import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)
from stable_baselines3 import SAC

from buoyantboat.env import \
    BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402

# Load the data from the CSV file
df = pd.read_csv("./buoyantboat/validate/ObjectState.csv")

z_array = df["z"].values
z_min = np.min(z_array)
z_max = np.max(z_array)
z_normalized = 2 * (z_array - z_min) / (z_max - z_min) - 1

model_path = os.path.abspath(
    "./trained_policies/converged/SAC_BEST_SO_FAR/best_model.zip"
)
model = SAC.load(model_path)

env = BuoyantBoat(
    control_technique="SAC",
    # target_velocity=0.0,
    target_position=25,
    max_step_per_episode=2000,
    learning_starts=0,
    validation=True,
)

obs, info = env.reset()
state_log = []
reward_log = []
action_log = []
preset_velocity = []
preset_position = []
winch_velocity = []
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, done, truncated, info = env.step(action=action, ext_pos=z_normalized)

    action_log.append(action[0])
    state_log.append(obs)
    reward_log.append(reward)
    preset_position.append(env.target_position)
    preset_velocity.append(env.target_velocity)
    winch_velocity.append(info["winch_velocity"])

state_log = np.array(state_log)
boat_pos = state_log[:, 4]
boat_vel = state_log[:, 5]
load_pos = state_log[:, 0]
load_vel = state_log[:, 2]


fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs[0, 0].plot(boat_pos, label="")
axs[0, 0].set_title("State 3 (boat_pos) over Time")
axs[0, 0].set_xlabel("Time step")
axs[0, 0].set_ylabel("State 3 (boat_pos)  value")

axs[0, 1].plot(load_pos, label="Load position")
axs[0, 1].plot(boat_pos, label="Boat position")
axs[0, 1].plot(preset_position, label="Target load position")
axs[0, 1].set_title("Positions over Time")
axs[0, 1].set_xlabel("Time step")
axs[0, 1].set_ylabel("State 1 (load_pos)  value")
axs[0, 1].legend()

axs[1, 1].plot(load_vel, label="Load velocity")
axs[1, 1].plot(boat_vel, label="Boat velocity")
axs[1, 1].plot(preset_velocity, label="Target load velocity")
axs[1, 1].set_title("Velocities over Time")
axs[1, 1].set_xlabel("Time step")
axs[1, 1].set_ylabel("State 1 (load_pos)  value")
axs[1, 1].legend()

axs[1, 0].plot(load_pos)
axs[1, 0].set_title("State 1 (load_pos) over Time")
axs[1, 0].set_xlabel("Time step")
axs[1, 0].set_ylabel("State 1 (load_pos)  value")

axs[2, 0].plot(action_log, label="Action")
axs[2, 0].plot(winch_velocity, label="Winch Velocity")
axs[2, 0].set_title("Action over Time")
axs[2, 0].set_xlabel("Time step")
axs[2, 0].set_ylabel("Action  value")
axs[2, 0].legend()

axs[2, 1].plot(reward_log)
axs[2, 1].set_title("Rewards over Time")
axs[2, 1].set_xlabel("Time step")
axs[2, 1].set_ylabel("Reward")

# Create a DataFrame from the state data
data = {
    "Boat_Position_Z": boat_pos,
    "Boat_Velocity_Z": boat_vel,
    "Load_Position_Z": load_pos,
    "Load_Velocity_Z": load_vel,
    "Target_Load_Position_Z": preset_position,
    "Target_Load_Velocity_Z": preset_velocity,
    "Action": action_log,
    "Winch_Velocity": winch_velocity,
    "Reward": reward_log,
}
df = pd.DataFrame(data)
# df.to_csv("./buoyantboat/validate/states_sac_no_winch_ue_data.csv", index=False)
plt.legend()
plt.tight_layout()
plt.show()
