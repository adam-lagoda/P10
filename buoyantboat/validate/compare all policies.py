import matplotlib.pyplot as plt
import numpy as np
import os 
import sys
package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)
from buoyantboat.env import BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402
from stable_baselines3 import SAC, DDPG, PPO


# model_path = os.path.abspath("./good_models/sac_position_velocity_linear_reward_full_optimized_for_25/best_model.zip")
# model_path = os.path.abspath(r"C:\Users\ADAM\OneDrive - Aalborg Universitet\P9\model\P10\boat_heave_comp_SAC_policy.zip")
model_path_ddpg = os.path.abspath("./good_models/DDPG_ALL_STATES_vel_pos_rewards/best_model.zip")
model_path_ppo = os.path.abspath("./good_models/PPO_ALL_STATES_vel_pos_rewards/best_model.zip")
model_path_sac = os.path.abspath("./good_models/SAC_ALL_STATES_vel_pos_reward/best_model.zip")
# model_path = os.path.abspath("./boat_heave_comp_SAC_policy.zip")
model_ddpg = DDPG.load(model_path_ddpg)
model_ppo = PPO.load(model_path_ppo)
model_sac = SAC.load(model_path_sac)

TARGET_VELOCITY_1 = 0
TARGET_VELOCITY_2 = 0
TARGET_POSITION_1 = 20
TARGET_POSITION_2 = 10

env_ppo = BuoyantBoat(
    control_technique="SAC",
    target_velocity=TARGET_VELOCITY_1,
    target_position=TARGET_POSITION_1,
    max_step_per_episode=2000
)
env_sac = BuoyantBoat(
    control_technique="SAC",
    target_velocity=TARGET_VELOCITY_1,
    target_position=TARGET_POSITION_1,
    max_step_per_episode=2000
)
env_ddpg = BuoyantBoat(
    control_technique="SAC",
    target_velocity=TARGET_VELOCITY_1,
    target_position=TARGET_POSITION_1,
    max_step_per_episode=2000
)

obs, info = env_ppo.reset()
state_log = []
reward_log_ppo = []
action_log_ppo = []
preset_velocity = []
preset_position = []
done = False

while not done:
    if env_ppo.step_count>1000:
        env_ppo.target_position=TARGET_POSITION_2
        env_ppo.target_velocity=TARGET_VELOCITY_2
    action, _states = model_ppo.predict(obs, deterministic=True)
    # action = [0.0]
    obs, reward, done, truncated, info = env_ppo.step(action)
    action_log_ppo.append(action[0])
    state_log.append(obs)
    reward_log_ppo.append(reward)
    preset_position.append(env_ppo.target_position)
    preset_velocity.append(env_ppo.target_velocity)

state_log = np.array(state_log)
boat_pos_ppo = state_log[:, 4]
boat_vel_ppo = state_log[:, 5]
load_pos_ppo = state_log[:, 0]
load_vel_ppo = state_log[:, 2]

obs, info = env_ddpg.reset()
state_log = []
reward_log_ddpg = []
action_log_ddpg = []
preset_velocity = []
preset_position = []
done = False

while not done:
    if env_ddpg.step_count>1000:
        env_ddpg.target_position=TARGET_POSITION_2
        env_ddpg.target_velocity=TARGET_VELOCITY_2
    action, _states = model_ddpg.predict(obs, deterministic=True)
    # action = [0.0]
    obs, reward, done, truncated, info = env_ddpg.step(action)
    action_log_ddpg.append(action[0])
    state_log.append(obs)
    reward_log_ddpg.append(reward)
    preset_position.append(env_ddpg.target_position)
    preset_velocity.append(env_ddpg.target_velocity)

state_log = np.array(state_log)
boat_pos_ddpg = state_log[:, 4]
boat_vel_ddpg = state_log[:, 5]
load_pos_ddpg = state_log[:, 0]
load_vel_ddpg = state_log[:, 2]

obs, info = env_sac.reset()
state_log = []
reward_log_sac = []
action_log_sac = []
preset_velocity = []
preset_position = []
done = False

while not done:
    if env_sac.step_count>1000:
        env_sac.target_position=TARGET_POSITION_2
        env_sac.target_velocity=TARGET_VELOCITY_2
    action, _states = model_sac.predict(obs, deterministic=True)
    # action = [0.0]
    obs, reward, done, truncated, info = env_sac.step(action)
    action_log_sac.append(action[0])
    state_log.append(obs)
    reward_log_sac.append(reward)
    preset_position.append(env_sac.target_position)
    preset_velocity.append(env_sac.target_velocity)

state_log = np.array(state_log)
boat_pos_sac = state_log[:, 4]
boat_vel_sac = state_log[:, 5]
load_pos_sac = state_log[:, 0]
load_vel_sac = state_log[:, 2]


fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs[0,0].plot(boat_pos_sac, label="sac")
axs[0,0].plot(boat_pos_ddpg, label="ddpg")
axs[0,0].plot(boat_pos_ppo, label="ppo")
axs[0,0].set_title("State 3 (boat_pos) over Time")
axs[0,0].set_xlabel("Time step")
axs[0,0].set_ylabel("State 3 (boat_pos)  value")

axs[0,1].plot(load_pos_ppo, label="Load position_ppo")
axs[0,1].plot(boat_pos_ppo, label="Boat position_ppo")
axs[0,1].plot(load_pos_ddpg, label="Load position_ddpg")
axs[0,1].plot(boat_pos_ddpg, label="Boat position_ddpg")
axs[0,1].plot(load_pos_sac, label="Load position_sac")
axs[0,1].plot(boat_pos_sac, label="Boat position_sac")
axs[0,1].plot(preset_position, label="Target load position")
axs[0,1].set_title("Positions over Time")
axs[0,1].set_xlabel("Time step")
axs[0,1].set_ylabel("State 1 (load_pos)  value")
axs[0,1].legend()

axs[1,1].plot(load_vel_ddpg, label="Load velocity_load_ddpg")
axs[1,1].plot(boat_vel_ddpg, label="Boat velocity_ddpg")
axs[1,1].plot(load_vel_sac, label="Load velocity_sac")
axs[1,1].plot(boat_vel_sac, label="Boat velocity_sac")
axs[1,1].plot(load_vel_ppo, label="Load velocity_ppo")
axs[1,1].plot(boat_vel_ppo, label="Boat velocity_ppo")
axs[1,1].plot(preset_velocity, label="Target load velocity")
axs[1,1].set_title("Velocities over Time")
axs[1,1].set_xlabel("Time step")
axs[1,1].set_ylabel("State 1 (load_pos)  value")
axs[1,1].legend()

axs[1,0].plot(load_pos_sac, label="sac")
axs[1,0].plot(load_pos_ddpg, label="ddpg")
axs[1,0].plot(load_pos_ppo, label="ppo")
axs[1,0].set_title("State 1 (load_pos) over Time")
axs[1,0].set_xlabel("Time step")
axs[1,0].set_ylabel("State 1 (load_pos)  value")
axs[1,0].legend()

axs[2,0].plot(action_log_ppo, label="ppo")
axs[2,0].plot(action_log_sac, label="sac")
axs[2,0].plot(action_log_ddpg, label="ddpg")
axs[2,0].set_title("Action over Time")
axs[2,0].set_xlabel("Time step")
axs[2,0].set_ylabel("Action  value")
axs[2,0].legend()

axs[2,1].plot(reward_log_ppo, label="ppo")
axs[2,1].plot(reward_log_sac, label="sac")
axs[2,1].plot(reward_log_ddpg, label="ddpg")
axs[2,1].set_title("Rewards over Time")
axs[2,1].set_xlabel("Time step")
axs[2,1].set_ylabel("Reward")
axs[2,1].legend()

plt.legend()
plt.tight_layout()
plt.show()
