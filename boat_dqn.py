from boat_env_cleared import BuoyantBoat
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from time import time

boat = BuoyantBoat(control_technique="DQN")
env = Monitor(boat)
# Initialize RL algorithm type and hyperparameters
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=8,  # 32
    train_freq=4,
    target_update_interval=200,
    learning_starts=250,
    buffer_size=100000,  # 500000
    max_grad_norm=10,
    exploration_fraction=0.5,
    exploration_final_eps=0.01,
    device="cpu",
    tensorboard_log="./tb_logs/",
)

# Create an evaluation callback with the same env, called every 10000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=10000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(total_timesteps=10000, tb_log_name="boat_heave", **kwargs)

# Save policy weights
model.save("boat_heave_comp_policy")
