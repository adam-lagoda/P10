import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)
from buoyantboat.env import BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402
from stable_baselines3 import SAC


model_path = os.path.abspath("./best_model.zip")
# model_path = os.path.abspath("./boat_heave_comp_SAC_policy.zip")
model = SAC.load(model_path)

env = BuoyantBoat(control_technique="SAC")

obs, info = env.reset()
state_log_no_control = []
reward_log_no_control = []
action_log_no_control = []
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    action = [0.0]
    # obs, reward, done, truncated, info = env.step([0.0])
    obs, reward, done, truncated, info = env.step(action)
    state_log_no_control.append(action[0])
    reward_log_no_control.append(obs)
    action_log_no_control.append(reward)

# model_path = os.path.abspath("./best_model.zip")
# # model_path = os.path.abspath("./boat_heave_comp_SAC_policy.zip")
# model = SAC.load(model_path)

# env = BuoyantBoat(control_technique="SAC")

# # Reset environment and model
# obs, info = env.reset()
# state_log_control = []
# reward_log_control = []
# action_log_control = []
# done = False

# while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     # obs, reward, done, truncated, info = env.step([0.0])
#     obs, reward, done, truncated, info = env.step(action)
#     state_log_control.append(action[0])
#     reward_log_control.append(obs)
#     action_log_control.append(reward)

state_log_no_control = np.array(state_log_no_control)
state_log_control = np.array(state_log_no_control)
# state_log_control = np.array(state_log_control)

fig, axs = plt.subplots(3, 2, figsize=(10, 10))
print("Shape of state_log_no_control:", state_log_no_control.shape)
print("Shape of state_log_control:", state_log_control.shape)
axs[0,0].plot(state_log_no_control[:, 2], label="no control")
axs[0,0].plot(state_log_control[:, 2], label="control")
axs[0,0].set_title("State 3 (boat_pos) over Time")
axs[0,0].set_xlabel("Time step")
axs[0,0].set_ylabel("State 3 (boat_pos)  value")
axs[0,0].legend()

axs[0,1].plot(state_log_no_control[:, 0], label="Load position - no control")
axs[0,1].plot(state_log_control[:, 0], label="Load position - control")
axs[0,1].plot(state_log_no_control[:, 2], label="Boat position - no control")
axs[0,1].plot(state_log_control[:, 2], label="Boat position - control")
axs[0,1].set_title("Positions over Time")
axs[0,1].set_xlabel("Time step")
axs[0,1].set_ylabel("State 1 (load_pos)  value")
axs[0,1].legend()

axs[1,1].plot(state_log_no_control[:, 1], label="Load velocity - no control")
axs[1,1].plot(state_log_control[:, 3], label="Boat velocity - control")
axs[1,1].plot(state_log_no_control[:, 1], label="Load velocity - no control")
axs[1,1].plot(state_log_control[:, 3], label="Boat velocity - control")
axs[1,1].set_title("Velocities over Time")
axs[1,1].set_xlabel("Time step")
axs[1,1].set_ylabel("State 1 (load_pos)  value")
axs[1,1].legend()

axs[1,0].plot(state_log_no_control[:, 0], label="no control")
axs[1,0].plot(state_log_control[:, 0], label="control")
axs[1,0].set_title("State 1 (load_pos) over Time")
axs[1,0].set_xlabel("Time step")
axs[1,0].set_ylabel("State 1 (load_pos)  value")
axs[1,0].legend()

axs[2,0].plot(state_log_no_control, label="no control")
axs[2,0].plot(state_log_control, label="control")
axs[2,0].set_title("Action over Time")
axs[2,0].set_xlabel("Time step")
axs[2,0].set_ylabel("Action  value")
axs[2,0].legend()

axs[2,1].plot(reward_log_no_control, label="no control")
axs[2,1].plot(reward_log_control, label="control")
axs[2,1].set_title("Rewards over Time")
axs[2,1].set_xlabel("Time step")
axs[2,1].set_ylabel("Reward")
axs[2,1].legend()

plt.legend()
plt.tight_layout()
plt.show()
