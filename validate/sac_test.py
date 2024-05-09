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


# model_path = os.path.abspath("./good_models/sac_position_velocity_linear_reward_full_optimized_for_25/best_model.zip")
model_path = os.path.abspath("./best_model.zip")
# model_path = os.path.abspath("./boat_heave_comp_SAC_policy.zip")
model = SAC.load(model_path)

env = BuoyantBoat(
    control_technique="SAC",
    target_velocity=0,
    target_position=15
)

obs, info = env.reset()
state_log = []
reward_log = []
action_log = []
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    # action = [0.0]
    obs, reward, done, truncated, info = env.step([0.0])
    obs, reward, done, truncated, info = env.step(action)
    action_log.append(action[0])
    state_log.append(obs)
    reward_log.append(reward)

state_log = np.array(state_log)

fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs[0,0].plot(state_log[:, 2], label="")
axs[0,0].set_title("State 3 (boat_pos) over Time")
axs[0,0].set_xlabel("Time step")
axs[0,0].set_ylabel("State 3 (boat_pos)  value")

axs[0,1].plot(state_log[:, 0], label="Load position")
axs[0,1].plot(state_log[:, 2], label="Boat position")
axs[0,1].set_title("Positions over Time")
axs[0,1].set_xlabel("Time step")
axs[0,1].set_ylabel("State 1 (load_pos)  value")
axs[0,1].legend()

axs[1,1].plot(state_log[:, 1], label="Load velocity")
axs[1,1].plot(state_log[:, 3], label="Boat velocity")
axs[1,1].set_title("Velocities over Time")
axs[1,1].set_xlabel("Time step")
axs[1,1].set_ylabel("State 1 (load_pos)  value")
axs[1,1].legend()

axs[1,0].plot(state_log[:, 0], label="")
axs[1,0].set_title("State 1 (load_pos) over Time")
axs[1,0].set_xlabel("Time step")
axs[1,0].set_ylabel("State 1 (load_pos)  value")

axs[2,0].plot(action_log)
axs[2,0].set_title("Action over Time")
axs[2,0].set_xlabel("Time step")
axs[2,0].set_ylabel("Action  value")

axs[2,1].plot(reward_log)
axs[2,1].set_title("Rewards over Time")
axs[2,1].set_xlabel("Time step")
axs[2,1].set_ylabel("Reward")

plt.legend()
plt.tight_layout()
plt.show()
