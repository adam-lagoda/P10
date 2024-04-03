import time

from boat_env_cleared import BuoyantBoat
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage


class ProgressBarCallback(BaseCallback):
    """
    A custom callback to print a progress bar for each training epoch.
    """

    def __init__(self, check_freq: int, total_timesteps: int, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.total_timesteps = total_timesteps
        self.start_time = time.time()

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        percent_complete = 100 * (current_step / self.total_timesteps)
        progress_str = f"Progress: {percent_complete:.1f}%"
        elapsed_time = time.time() - self.start_time
        print(f"\r{progress_str} - Elapsed Time: {elapsed_time:.2f}s", end="")
        if current_step % self.check_freq == 0:
            print("")  # New line for clean separation
        return True


boat = BuoyantBoat(control_technique="DQN")

# Initialize DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=8,  # 32
    train_freq=4,
    target_update_interval=10000,
    learning_starts=250,
    buffer_size=100000,  # 500000
    max_grad_norm=10,
    exploration_fraction=0.5,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)

# Initialize evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/best_model",
    n_eval_episodes=5,
    eval_freq=10000,
    log_path="./logs/results",
)

# Initialize the progress bar callback
progress_bar_callback = ProgressBarCallback(check_freq=1000, total_timesteps=10000, verbose=1)

# Train the model with callbacks
model.learn(total_timesteps=10000, tb_log_name="Fog_Conf_test_0.75", callback=[eval_callback, progress_bar_callback])

# Save the model
model.save("dqn_boat_policy")
