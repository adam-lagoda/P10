"""File contatining BuoyantBoat class that holds dynamics of a boat."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)
from buoyantboat.env import BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402

if __name__ == "__main__":

    env = BuoyantBoat(
        control_technique="SAC",
        max_step_per_episode=10000
    )
    # env.reset()
    # env.step(0)
    _ = env.reset()

    state_positions = []
    state_orientations = []
    state_angular_velocities = []
    state_load_positions = []
    _current_buyoancy_forces = []
    state_load_velocities = []
    wave_data = []

    for i in tqdm(range(10000)):
        # Take a step in the environment
        state, _, _, _, _ = env.step([0.1])  # Assuming action 5 is used for all steps

        # Temporary variable to hold current state values
        state_positions.append(state[0])
        state_orientations.append(state[2])
        state_angular_velocities.append(state[3])
        state_load_positions.append(state[4])
        wave_data.append(state[5])
        state_load_velocities.append(state[6])
        _current_buyoancy_forces.append(state[7])

    # Convert lists to numpy arrays for plotting
    np_positions = np.array(state_positions)
    np_orientations = np.array(state_orientations)
    np_angular_velocities = np.array(state_angular_velocities)
    np_load_position = np.array(state_load_positions)
    np_current_buyoancy_forces = np.array(_current_buyoancy_forces)
    np_state_load_velocities = np.array(state_load_velocities)
    np_wave_data = np.array(wave_data)

    # Plot states over time
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    # plt.plot(positions[:, 0], label='Position X')
    plt.plot(np_load_position[:, 2], label="Position load")
    plt.plot(np_positions[:, 2], label="Position Boat CM")
    plt.plot(np_wave_data, label="Wave height")
    plt.title(f"Position over time, 1ts = {env.dt}s")
    plt.legend()

    plt.subplot(4, 1, 2)
    # plt.plot(positions[:, 0], label='Position X')
    plt.plot(np_load_position[:, 2], label="Position load")
    # plt.plot(np_positions[:, 2], label="Position Boat CM")
    # plt.plot(np_wave_data, label="Wave height")
    plt.title(f"Position over time, 1ts = {env.dt}s")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(np_orientations[:, 0], label="Orientation Roll")
    plt.plot(np_orientations[:, 1], label="Orientation Pitch")
    plt.plot(np_orientations[:, 2], label="Orientation Yaw")
    plt.title(f"Orientation over time, 1ts = {env.dt}s")
    plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(np_angular_velocities[:, 0], label='Angular Velocity Roll')
    # plt.plot(np_angular_velocities[:, 1], label='Angular Velocity Pitch')
    # plt.plot(np_angular_velocities[:, 2], label='Angular Velocity Yaw')
    # plt.title('Angular Velocity over time')
    # plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(np_current_buyoancy_forces[:, 0, 0], label="buoyant force #1")
    plt.plot(np_current_buyoancy_forces[:, 1, 0], label="buoyant force #2")
    plt.plot(np_current_buyoancy_forces[:, 0, 1], label="buoyant force #3")
    plt.plot(np_current_buyoancy_forces[:, 1, 1], label="buoyant force #4")
    plt.title(f"buoyant forces,  1ts = {env.dt}s")
    plt.legend()

    plt.tight_layout()
    plt.show()
