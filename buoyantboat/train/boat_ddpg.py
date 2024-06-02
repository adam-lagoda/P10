import os
import sys

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.ddpg import MlpPolicy

package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)
from buoyantboat.env import \
    BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402

boat = BuoyantBoat(control_technique="SAC")
env = Monitor(boat)


# Initialize RL algorithm type and hyperparameters
model = DDPG(
    MlpPolicy,
    env,
    learning_rate=0.3,
    buffer_size=100000,
    learning_starts=10,
    batch_size=100,
    train_freq=(1, "episode"),
    gradient_steps=-1,  # Update policy after every episode
    tensorboard_log="./tb_logs_ddpg/",
    device="cuda",
    verbose=1,
)

# Create an evaluation callback with the same env, called every 5000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=5000,
)
callbacks.append(eval_callback)

kwargs = {}
kwargs["callback"] = callbacks

# Train for a certain number of timesteps
model.learn(
    total_timesteps=200000,
    tb_log_name="boat_heave_ddpg_new",
    progress_bar=True,
    **kwargs
)

# Save policy weights
model.save("boat_heave_comp_DDPG_new")
