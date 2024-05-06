import matplotlib.pyplot as plt
import numpy as np
from buoyantboat.boat_env import BuoyantBoat  # Assuming this is where your environment class is defined
from stable_baselines3 import DQN

# Load the trained model
model_path = "boat_heave_comp_policy.zip"
model = DQN.load(model_path)

# Load your custom environment
env = BuoyantBoat(control_technique="DQN")

# Reset the environment and initialize variables for plotting
obs = env.reset()
state_log = []
reward_log = []
done = False

# Run the policy for a single episode and collect data
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    state_log.append(obs)
    reward_log.append(reward)

# Convert the states log to a numpy array for easy plotting (if it is not already an array)
state_log = np.array(state_log)

# Create subplots for each state dimension
fig, axs = plt.subplots(state_log.shape[1] + 1, figsize=(10, 10))

axs[0].plot(state_log[:, 0])
axs[0].set_title("State 0 (boat_pos) over Time")
axs[0].set_xlabel("Time step")
axs[0].set_ylabel("State 0 (boat_pos)  value")

axs[1].plot(state_log[:, 1])
axs[1].set_title("State 1 (load_pos) over Time")
axs[1].set_xlabel("Time step")
axs[1].set_ylabel("State 1 (load_pos)  value")

axs[2].plot(reward_log)
axs[2].set_title("Rewards over Time")
axs[2].set_xlabel("Time step")
axs[2].set_ylabel("Reward")

# Add a legend and show the plot
plt.legend()
plt.tight_layout()
plt.show()
