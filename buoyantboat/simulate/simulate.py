"""File contatining BuoyantBoat class that holds dynamics of a boat."""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

package_path = os.path.abspath(os.path.join(os.getcwd()))
package_path = package_path[0].upper() + package_path[1:]
if package_path not in sys.path:
    sys.path.append(package_path)
from buoyantboat.env import \
    BuoyantBoat  # pylint: disable=wrong-import-position; noqa: E402

if __name__ == "__main__":

    env = BuoyantBoat(
        control_technique="SAC",
        max_step_per_episode=10000,
        # learning_starts=8000,
        learning_starts=1,
        validation=True,
    )
    _ = env.reset()

    state_positions = []
    state_orientations = []
    state_angular_velocities = []
    state_load_positions = []
    _current_buyoancy_forces = []
    state_load_velocities = []
    wave_data = []

    for i in tqdm(range(10000)):
        state, _, done, _, _ = env.step(
            [0.00]
        )  # Assuming action 5 is used for all steps
        if done:
            break
        state_positions.append(state[0])
        state_orientations.append(state[2])
        state_angular_velocities.append(state[3])
        state_load_positions.append(state[4])
        wave_data.append(state[5])
        state_load_velocities.append(state[6])
        _current_buyoancy_forces.append(state[7])

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
    plt.plot(np_load_position[:, 2], label="Position load")
    plt.plot(np_positions[:, 2], label="Position Boat CM")
    plt.plot(np_wave_data, label="Wave height")
    plt.title(f"Position over time, 1ts = {env.dt}s")
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(np_load_position[:, 2], label="Position load")
    plt.title(f"Position over time, 1ts = {env.dt}s")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(np_orientations[:, 0], label="Orientation Roll")
    plt.plot(np_orientations[:, 1], label="Orientation Pitch")
    plt.plot(np_orientations[:, 2], label="Orientation Yaw")
    plt.title(f"Orientation over time, 1ts = {env.dt}s")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(np_current_buyoancy_forces[:, 0, 0], label="buoyant force #1")
    plt.plot(np_current_buyoancy_forces[:, 1, 0], label="buoyant force #2")
    plt.plot(np_current_buyoancy_forces[:, 0, 1], label="buoyant force #3")
    plt.plot(np_current_buyoancy_forces[:, 1, 1], label="buoyant force #4")
    plt.title(f"buoyant forces,  1ts = {env.dt}s")
    plt.legend()

    # Create a DataFrame from the state data
    data = {
        "Position_X": np_positions[:, 0],
        "Position_Y": np_positions[:, 1],
        "Position_Z": np_positions[:, 2],
        "Orientation_Roll": np_orientations[:, 0],
        "Orientation_Pitch": np_orientations[:, 1],
        "Orientation_Yaw": np_orientations[:, 2],
        "Angular_Velocity_Roll": np_angular_velocities[:, 0],
        "Angular_Velocity_Pitch": np_angular_velocities[:, 1],
        "Angular_Velocity_Yaw": np_angular_velocities[:, 2],
        "Load_Position_X": np_load_position[:, 0],
        "Load_Position_Y": np_load_position[:, 1],
        "Load_Position_Z": np_load_position[:, 2],
        "Wave_Height": np_wave_data,
        "Load_Velocity_Z": np_state_load_velocities,
        "Buoyant_Force_1_X": np_current_buyoancy_forces[:, 0, 0],
        "Buoyant_Force_1_Y": np_current_buyoancy_forces[:, 0, 1],
        "Buoyant_Force_2_X": np_current_buyoancy_forces[:, 1, 0],
        "Buoyant_Force_2_Y": np_current_buyoancy_forces[:, 1, 1],
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    # df.to_csv("states_no_control_no_wait.csv", index=False)

    plt.tight_layout()
    plt.show()
