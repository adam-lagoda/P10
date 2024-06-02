import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.ppo import MlpPolicy

package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)

from buoyantboat.env import \
    BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402

boat = BuoyantBoat(control_technique="SAC")
env = Monitor(boat)

# Initialize RL algorithm type and hyperparameters
model = PPO(
    MlpPolicy,
    env,
    learning_rate=0.003,
    verbose=1,
    batch_size=64,
    tensorboard_log="./tb_logs_ppo/",
    device="cuda",
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
model.learn(
    total_timesteps=200000, tb_log_name="boat_heave_ppo", progress_bar=True, **kwargs
)

# Save policy weights
model.save("boat_heave_comp_PPO_policy_new")
