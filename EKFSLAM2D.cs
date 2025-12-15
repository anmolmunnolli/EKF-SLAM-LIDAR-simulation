using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class EKFSLAM2D : MonoBehaviour
{
    [Header("Robot (true)")]
    public Transform robot;
    public RobotOdometry odom;

    [Header("Trajectory Rendering")]
    public LineRenderer estPathRenderer;
    public LineRenderer truePathRenderer;

    [Header("Trajectories")]
    public LineRenderer truePath;
    public LineRenderer estPath;

    public int maxTrailPoints = 500;

    private List<Vector3> trueTrail = new List<Vector3>();
    private List<Vector3> estTrail = new List<Vector3>();

    List<float> kalmanGainNorm = new List<float>();
    bool logKalmanGain = true;

    List<float> timeLog = new();
    List<float> msePoseLog = new();
    List<float> mseLandmarkLog = new();

    [Header("Heading alignment (IMPORTANT)")]
    [Tooltip("If your sprite forward direction is not +X, set a constant offset (deg). Example: 90 or -90.")]
    public float headingOffsetDeg = 0f;

    [Header("Landmarks (true)")]
    public Landmark[] landmarkObjs;

    [Header("Sensor (LiDAR-like)")]
    public float maxRange = 20f;
    public float fovDeg = 360f;
    public float measRateHz = 15f;

    [Tooltip("Measurement noise (meters)")]
    public float sigmaRange = 0.25f;

    [Tooltip("Measurement noise (degrees)")]
    public float sigmaBearingDeg = 3.0f;

    [Header("Motion Noise (EKF prediction)")]
    [Tooltip("Process noise on linear velocity (m/s)")]
    public float sigmaV = 0.20f;

    [Tooltip("Process noise on angular velocity (deg/s)")]
    public float sigmaWDeg = 8.0f;

    [Header("Outlier Rejection (Gating)")]
    public bool enableGating = true;
    public float gateMaxDr = 2.0f;        // meters
    public float gateMaxDbDeg = 25.0f;    // degrees

    [Header("Visualization")]
    public GameObject estLandmarkPrefab;   // prefab ONLY, do not keep instances in scene
    public bool drawTrails = true;

    // EKF state: [xr, yr, theta, l1x,l1y,l2x,l2y,...]
    private Vector<float> mu;
    private Matrix P;

    private Dictionary<int, int> lmIndex = new Dictionary<int, int>();          // id -> state start index
    private Dictionary<int, Transform> estLmViz = new Dictionary<int, Transform>(); // id -> visual transform

    private float measTimer;

    // private readonly List<Vector3> estTrail = new List<Vector3>();
    // private readonly List<Vector3> trueTrail = new List<Vector3>();

    void Start()
    {
        if (robot == null) { Debug.LogError("Assign robot"); enabled = false; return; }
        if (odom == null) odom = robot.GetComponent<RobotOdometry>();
        if (odom == null) { Debug.LogError("RobotOdometry missing on robot"); enabled = false; return; }

        if (landmarkObjs == null || landmarkObjs.Length == 0)
            landmarkObjs = FindObjectsByType<Landmark>(FindObjectsSortMode.None);

        // Initial robot state guess (deliberately noisy)
        mu = new Vector<float>(3);
        mu[0] = robot.position.x + Random.Range(-0.5f, 0.5f);
        mu[1] = robot.position.y + Random.Range(-0.5f, 0.5f);
        mu[2] = WrapPi(GetRobotHeadingRad() + Random.Range(-0.2f, 0.2f));

        // Initial covariance: not too small (avoid overconfidence)
        P = Matrix.Identity(3);
        P[0,0] = 1.0f;
        P[1,1] = 1.0f;
        P[2,2] = 0.5f;

        estPathRenderer.positionCount = 0;
        estPathRenderer.startWidth = 0.05f;
        estPathRenderer.endWidth = 0.05f;
        estPathRenderer.material = new Material(Shader.Find("Sprites/Default"));
        estPathRenderer.startColor = Color.cyan;
        estPathRenderer.endColor = Color.cyan;

        truePathRenderer.positionCount = 0;
        truePathRenderer.startWidth = 0.05f;
        truePathRenderer.endWidth = 0.05f;
        truePathRenderer.material = new Material(Shader.Find("Sprites/Default"));
        truePathRenderer.startColor = Color.green;
        truePathRenderer.endColor = Color.green;

    }

    void Update()
    {
        float dt = Time.deltaTime;

        // 1) EKF Predict
        Predict(dt);

        // 2) Measurements
        measTimer += dt;
        float period = 1f / Mathf.Max(1f, measRateHz);
        while (measTimer >= period)
        {
            measTimer -= period;
            var measurements = SenseLandmarks();
            foreach (var z in measurements)
                UpdateEKF(z.id, z.range, z.bearing);
        }

        // 3) Store trajectories
        if (drawTrails)
        {
            trueTrail.Add(robot.position);
            estTrail.Add(new Vector3(mu[0], mu[1], 0f));

            if (trueTrail.Count > maxTrailPoints)
                trueTrail.RemoveAt(0);

            if (estTrail.Count > maxTrailPoints)
                estTrail.RemoveAt(0);

            // Push into LineRenderers
            truePath.positionCount = trueTrail.Count;
            truePath.SetPositions(trueTrail.ToArray());

            estPath.positionCount = estTrail.Count;
            estPath.SetPositions(estTrail.ToArray());
        }
        
        // Robot pose MSE
        float poseMSE =
            Mathf.Pow(mu[0] - robot.position.x, 2) +
            Mathf.Pow(mu[1] - robot.position.y, 2);

        // Landmark MSE
        float lmMSE = 0f;
        int count = 0;

        foreach (var lm in landmarkObjs)
        {
            if (!lmIndex.ContainsKey(lm.id)) continue;

            int idx = lmIndex[lm.id];
            float dx = mu[idx]     - lm.transform.position.x;
            float dy = mu[idx + 1] - lm.transform.position.y;

            lmMSE += dx * dx + dy * dy;
            count++;
        }

        lmMSE /= Mathf.Max(1, count);

        // Log values
        timeLog.Add(Time.time);
        msePoseLog.Add(poseMSE);
        mseLandmarkLog.Add(lmMSE);

        UpdateLandmarkVisuals();

    }



    // ---------------- SENSOR ----------------
    struct Measurement { public int id; public float range; public float bearing; }

    List<Measurement> SenseLandmarks()
    {
        var list = new List<Measurement>();

        Vector2 rPos = new Vector2(robot.position.x, robot.position.y);
        float rTheta = GetRobotHeadingRad(); // IMPORTANT: heading must match motion!

        float halfFov = Mathf.Deg2Rad * (fovDeg * 0.5f);

        foreach (var lm in landmarkObjs)
        {
            Vector2 lPos = new Vector2(lm.transform.position.x, lm.transform.position.y);
            Vector2 d = lPos - rPos;

            float range = d.magnitude;
            if (range > maxRange) continue;

            float bearing = WrapPi(Mathf.Atan2(d.y, d.x) - rTheta);
            if (Mathf.Abs(bearing) > halfFov) continue;

            float noisyRange = range + Gaussian(0f, sigmaRange);
            float noisyBearing = WrapPi(bearing + Gaussian(0f, Mathf.Deg2Rad * sigmaBearingDeg));

            list.Add(new Measurement { id = lm.id, range = noisyRange, bearing = noisyBearing });
        }

        return list;
    }

    float GetRobotHeadingRad()
    {
        return WrapPi(Mathf.Deg2Rad * (robot.eulerAngles.z + headingOffsetDeg));
    }

    // ---------------- PREDICT ----------------
    void Predict(float dt)
    {
        float v = odom.v; // m/s
        float w = odom.w; // rad/s

        float x = mu[0];
        float y = mu[1];
        float th = mu[2];

        // Unicycle model
        mu[0] = x + v * Mathf.Cos(th) * dt;
        mu[1] = y + v * Mathf.Sin(th) * dt;
        mu[2] = WrapPi(th + w * dt);

        int n = mu.Length;

        // Jacobian F (state transition)
        Matrix F = Matrix.Identity(n);
        F[0,2] = -v * Mathf.Sin(th) * dt;
        F[1,2] =  v * Mathf.Cos(th) * dt;

        // Continuous-time-ish process noise scaled by dt
        float qv = sigmaV * sigmaV;
        float qw = (Mathf.Deg2Rad * sigmaWDeg) * (Mathf.Deg2Rad * sigmaWDeg);

        Matrix Q = Matrix.Zero(n, n);
        Q[0,0] = qv * dt;
        Q[1,1] = qv * dt;
        Q[2,2] = qw * dt;

        P = F * P * F.Transpose() + Q;
    }

    // ---------------- UPDATE ----------------
    void UpdateEKF(int id, float zRange, float zBearing)
    {
        if (!lmIndex.ContainsKey(id))
            InitializeLandmark(id, zRange, zBearing);

        int li = lmIndex[id];
        int n = mu.Length;

        float xr = mu[0], yr = mu[1], th = mu[2];
        float lx = mu[li], ly = mu[li + 1];

        float dx = lx - xr;
        float dy = ly - yr;

        float q = dx*dx + dy*dy;
        float rHat = Mathf.Sqrt(Mathf.Max(1e-9f, q));
        float bHat = WrapPi(Mathf.Atan2(dy, dx) - th);

        float dr = zRange - rHat;
        float db = WrapPi(zBearing - bHat);

        // Gating (reject crazy residuals)
        if (enableGating)
        {
            if (Mathf.Abs(dr) > gateMaxDr) return;
            if (Mathf.Abs(db) > Mathf.Deg2Rad * gateMaxDbDeg) return;
        }

        // R
        Matrix R = new Matrix(2,2);
        R[0,0] = sigmaRange * sigmaRange;
        float sb = Mathf.Deg2Rad * sigmaBearingDeg;
        R[1,1] = sb * sb;

        // H (2 x n)
        Matrix H = new Matrix(2, n);

        float rInv = 1f / rHat;
        float qInv = 1f / Mathf.Max(1e-9f, q);

        // range row
        H[0,0] = -dx * rInv;
        H[0,1] = -dy * rInv;
        H[0,2] = 0f;
        H[0,li]   = dx * rInv;
        H[0,li+1] = dy * rInv;

        // bearing row
        H[1,0] =  dy * qInv;
        H[1,1] = -dx * qInv;
        H[1,2] = -1f;
        H[1,li]   = -dy * qInv;
        H[1,li+1] =  dx * qInv;

        // S, K
        Matrix S = H * P * H.Transpose() + R;
        Matrix SInv = S.Inverse2x2();
        Matrix K = P * H.Transpose() * SInv;
        if (logKalmanGain)
        {
            float gainNorm = 0f;

            int measDim = 2; // range + bearing

            // Pose rows: x, y, theta = rows 0,1,2
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < measDim; j++)
                {
                    float v = K[i, j];
                    gainNorm += v * v;
                }
            }

            gainNorm = Mathf.Sqrt(gainNorm);
            kalmanGainNorm.Add(gainNorm);
        }



        // innovation
        Vector<float> innov = new Vector<float>(2);
        innov[0] = dr;
        innov[1] = db;

        mu = mu + (K * innov);
        mu[2] = WrapPi(mu[2]);

        // Joseph form covariance update (more stable):
        // P = (I-KH) P (I-KH)' + K R K'
        Matrix I = Matrix.Identity(n);
        Matrix KH = K * H;
        Matrix A = (I - KH);
        P = A * P * A.Transpose() + (K * R * K.Transpose());
    }

    void InitializeLandmark(int id, float zRange, float zBearing)
    {
        int oldN = mu.Length;
        int newN = oldN + 2;

        Vector<float> mu2 = new Vector<float>(newN);
        for (int i = 0; i < oldN; i++) mu2[i] = mu[i];

        float xr = mu[0], yr = mu[1], th = mu[2];
        float ang = th + zBearing;

        float lx = xr + zRange * Mathf.Cos(ang);
        float ly = yr + zRange * Mathf.Sin(ang);

        mu2[oldN] = lx;
        mu2[oldN+1] = ly;
        mu = mu2;

        Matrix P2 = Matrix.Zero(newN, newN);
        for (int r = 0; r < oldN; r++)
            for (int c = 0; c < oldN; c++)
                P2[r,c] = P[r,c];

        // Big initial uncertainty for new landmark (don’t be overconfident)
        P2[oldN, oldN] = 25f;
        P2[oldN+1, oldN+1] = 25f;

        P = P2;

        lmIndex[id] = oldN;

        if (estLandmarkPrefab != null && !estLmViz.ContainsKey(id))
        {
            var go = Instantiate(estLandmarkPrefab, new Vector3(lx, ly, 0f), Quaternion.identity);
            go.name = $"EST_LM_{id}";
            estLmViz[id] = go.transform;
        }
    }

    void UpdateLandmarkVisuals()
    {
        foreach (var kv in lmIndex)
        {
            int id = kv.Key;
            int li = kv.Value;

            if (estLmViz.TryGetValue(id, out Transform t))
                t.position = new Vector3(mu[li], mu[li + 1], 0f);
        }
    }

    // void OnDrawGizmos()
    // {
    //     if (!Application.isPlaying) return;

    //     // Estimated pose
    //     Gizmos.color = Color.cyan;
    //     Gizmos.DrawSphere(new Vector3(mu[0], mu[1], 0f), 0.08f);

    //     if (robot != null)
    //     {
    //         Gizmos.color = Color.green;
    //         Gizmos.DrawSphere(new Vector3(robot.position.x, robot.position.y, 0f), 0.06f);
    //     }

    //     if (drawTrails && estTrail.Count > 2)
    //     {
    //         Gizmos.color = Color.cyan;
    //         for (int i = 1; i < estTrail.Count; i++)
    //             Gizmos.DrawLine(estTrail[i-1], estTrail[i]);

    //         Gizmos.color = Color.green;
    //         for (int i = 1; i < trueTrail.Count; i++)
    //             Gizmos.DrawLine(trueTrail[i-1], trueTrail[i]);
    //     }
    // }

    // ---------------- utils ----------------
    static float WrapPi(float a)
    {
        while (a > Mathf.PI) a -= 2f * Mathf.PI;
        while (a < -Mathf.PI) a += 2f * Mathf.PI;
        return a;
    }

    static float Gaussian(float mean, float std)
    {
        float u1 = Mathf.Clamp(Random.value, 1e-6f, 1f);
        float u2 = Mathf.Clamp(Random.value, 1e-6f, 1f);
        float z0 = Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
        return mean + z0 * std;
    }
    void SaveKalmanGainCSV()
    {
        string path = Application.dataPath + "/kalman_gain.csv";
        using (var sw = new System.IO.StreamWriter(path))
        {
            sw.WriteLine("t,kalman_gain");
            for (int i = 0; i < kalmanGainNorm.Count; i++)
                sw.WriteLine($"{i},{kalmanGainNorm[i]}");
        }
        Debug.Log("Saved Kalman gain log to: " + path);
    }

    void OnApplicationQuit()
    {
        string path = System.IO.Path.Combine(Application.dataPath, "slam_mse.csv");
        SaveKalmanGainCSV();

        using StreamWriter sw = new StreamWriter(path);
        sw.WriteLine("time,mse_pose,mse_landmarks");

        for (int i = 0; i < timeLog.Count; i++)
        {
            sw.WriteLine($"{timeLog[i]},{msePoseLog[i]},{mseLandmarkLog[i]}");
        }

        Debug.Log($"Saved MSE log to: {path}");
    }

}
