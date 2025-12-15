using UnityEngine;

[RequireComponent(typeof(RobotOdometry))]
public class Path : MonoBehaviour
{
    [Header("Circular Path Parameters")]
    public Transform center;        // orbit center
    public float radius = 4f;       // meters
    public float angularSpeed = 0.4f; // rad/sec (NOT deg)

    [Header("Initial Heading")]
    public float initialHeadingDeg = 0f;

    // internal state
    private float theta;            // robot heading (rad)
    private float phi;              // angle around the circle (rad)

    private RobotOdometry odom;

    void Start()
    {
        odom = GetComponent<RobotOdometry>();

        // initialize pose relative to center
        Vector2 offset = transform.position - center.position;
        phi = Mathf.Atan2(offset.y, offset.x);

        theta = initialHeadingDeg * Mathf.Deg2Rad;

        // force correct initial rotation
        transform.rotation = Quaternion.Euler(0, 0, theta * Mathf.Rad2Deg);
    }

    void Update()
    {
        float dt = Time.deltaTime;

        // ---- COMMAND VELOCITIES (THIS IS THE KEY) ----
        float w = angularSpeed;           // rad/sec
        float v = radius * angularSpeed;  // m/sec

        // publish to odometry (EKF reads this)
        odom.v = v;
        odom.w = w;

        // ---- INTEGRATE MOTION (SAME MODEL AS EKF) ----
        theta += w * dt;
        theta = WrapPi(theta);

        float dx = v * Mathf.Cos(theta) * dt;
        float dy = v * Mathf.Sin(theta) * dt;

        transform.position += new Vector3(dx, dy, 0f);
        transform.rotation = Quaternion.Euler(0, 0, theta * Mathf.Rad2Deg);
    }

    static float WrapPi(float a)
    {
        while (a > Mathf.PI) a -= 2f * Mathf.PI;
        while (a < -Mathf.PI) a += 2f * Mathf.PI;
        return a;
    }
}
