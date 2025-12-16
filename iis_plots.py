import pandas as pd
import matplotlib.pyplot as plt

# Load CSVs
mse_df = pd.read_csv("slam_mse.csv")
kg_df  = pd.read_csv("kalman_gain.csv")

# Rolling window
window = 100
mse_df["pose_mse_smooth"] = mse_df["mse_pose"].rolling(window).mean()
mse_df["lm_mse_smooth"] = mse_df["mse_landmarks"].rolling(window).mean()
kg_df["kg_smooth"] = kg_df["kalman_gain"].rolling(window).mean()

fig, axes = plt.subplots(3, 1, figsize=(10, 11))

# ---- Plot 1: Pose MSE ----
axes[0].plot(mse_df["time"], mse_df["pose_mse_smooth"], color="blue")
axes[0].set_ylabel("Pose MSE")
axes[0].set_title("EKF-SLAM Pose Convergence")
axes[0].grid(True)

# ---- Plot 2: Landmark MSE ----
axes[1].plot(mse_df["time"], mse_df["lm_mse_smooth"], color="orange")
axes[1].set_ylabel("Landmark MSE")
axes[1].set_title("EKF-SLAM Landmark Convergence")
axes[1].grid(True)

# ---- Plot 3: Kalman Gain (different time scale) ----
axes[2].plot(kg_df["t"], kg_df["kg_smooth"], color="green")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Kalman Gain Norm")
axes[2].set_title("Kalman Gain Magnitude (High-Frequency Updates)")
axes[2].grid(True)

plt.tight_layout()
plt.show()
