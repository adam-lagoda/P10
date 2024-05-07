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
model = SAC.load(model_path)

env = BuoyantBoat(control_technique="SAC")

obs, info = env.reset()
state_log = []
reward_log = []
action_log = []
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    # obs, reward, done, truncated, info = env.step([500])
    obs, reward, done, truncated, info = env.step(action)
    action_log.append(action[0])
    state_log.append(obs)
    reward_log.append(reward)

state_log = np.array(state_log)

fig, axs = plt.subplots(state_log.shape[1] + 2, figsize=(10, 10))

axs[0].plot(state_log[:, 0])
axs[0].set_title("State 0 (boat_pos) over Time")
axs[0].set_xlabel("Time step")
axs[0].set_ylabel("State 0 (boat_pos)  value")

axs[1].plot(state_log[:, 1])
axs[1].set_title("State 1 (load_pos) over Time")
axs[1].set_xlabel("Time step")
axs[1].set_ylabel("State 1 (load_pos)  value")

axs[2].plot(action_log)
axs[2].set_title("Action over Time")
axs[2].set_xlabel("Time step")
axs[2].set_ylabel("Action  value")

axs[3].plot(reward_log)
axs[3].set_title("Rewards over Time")
axs[3].set_xlabel("Time step")
axs[3].set_ylabel("Reward")

plt.legend()
plt.tight_layout()
plt.show()
