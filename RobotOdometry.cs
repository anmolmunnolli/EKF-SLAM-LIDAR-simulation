using UnityEngine;

public class RobotOdometry : MonoBehaviour
{
    public float v; // m/s
    public float w; // rad/s

    private Vector3 prevPos;
    private float prevTheta;

    void Start()
    {
        prevPos = transform.position;
        prevTheta = transform.eulerAngles.z;
    }

    void Update()
    {
        float dt = Time.deltaTime;
        if (dt <= 1e-6f) return;

        Vector3 pos = transform.position;
        float theta = transform.eulerAngles.z;

        // linear velocity
        v = Vector3.Distance(pos, prevPos) / dt;

        // angular velocity (deg -> rad)
        float dthetaDeg = Mathf.DeltaAngle(prevTheta, theta);
        w = (dthetaDeg * Mathf.Deg2Rad) / dt;

        prevPos = pos;
        prevTheta = theta;
    }
}
