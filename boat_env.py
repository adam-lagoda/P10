import math
from typing import Literal
import csv
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm

class WaveGenerator:
    def __init__(
        self,
        coords,  # wrt to center of mass !!!
        frequency_1=0.0062,  # 0.1Hz
        frequency_2=0.0062,  # 0.1Hz
        num_points=101,
        timestep_delay=0.1,
    ):
        # Parameters
        self.frequency_1 = frequency_1  # frequency of 1st sine wave [rad/s]
        self.frequency_2 = frequency_2  # frequency of 2nd sine wave [rad/s]
        self.num_points = num_points  # number of points in each direction
        self.timestep_delay = timestep_delay  # delay in seconds for the second wave
        self.coords = coords
        # Grid of points
        self.x = np.linspace(-10, 10, self.num_points)
        self.y = np.linspace(-10, 10, self.num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self._wave_plane = self.wave1(self.X, 0) + self.wave2(self.Y, 0, self.timestep_delay)
        self.wave = None

    def _bilin_intp(self, index: tuple, arr=None):
        """Calculate a value based on non-int array index using bilinear interpolation."""
        if arr is None:
            arr = self._wave_plane
        # Get the integer parts of the indices
        x_floor = int(np.floor(index[0]))
        x_ceil = int(np.ceil(index[0]))
        y_floor = int(np.floor(index[1]))
        y_ceil = int(np.ceil(index[1]))

        # Ensure indices are within the array bounds
        x_floor = max(x_floor, 0)
        x_ceil = min(x_ceil, arr.shape[0] - 1)
        y_floor = max(y_floor, 0)
        y_ceil = min(y_ceil, arr.shape[1] - 1)

        # If the indices are already integers, no interpolation is needed
        if x_floor == x_ceil and y_floor == y_ceil:
            return arr[x_floor, y_floor]

        # Get the four surrounding points
        top_left = arr[x_floor, y_floor]
        top_right = arr[x_floor, y_ceil]
        bottom_left = arr[x_ceil, y_floor]
        bottom_right = arr[x_ceil, y_ceil]

        # Calculate the interpolation weights
        x_weight = index[0] - x_floor
        y_weight = index[1] - y_floor

        # Perform bilinear interpolation
        top_interpolated = (top_right * x_weight) + (top_left * (1 - x_weight))
        bottom_interpolated = (bottom_right * x_weight) + (bottom_left * (1 - x_weight))
        interpolated_value = (bottom_interpolated * y_weight) + (top_interpolated * (1 - y_weight))

        return interpolated_value

    def wave1(self, X, time):
        return 1 / 4 * np.sin(self.frequency_1 * (X + time * 2 * np.pi))

    def wave2(self, Y, time, delay):
        return 1 / 4 * np.sin(self.frequency_2 * (Y + (time - delay) * 2 * np.pi))

    def coordinates_to_indices(self, coordinates: tuple):
        index_x = int((coordinates[0] + self.x.min()) / np.round(self.x[1] - self.x[0], 1))
        index_y = int((coordinates[1] + self.y.min()) / np.round(self.y[1] - self.y[0], 1))
        return index_x, index_y

    def update(self, dt):
        self._wave_plane = self.wave1(self.X, dt) + self.wave2(self.Y, dt, self.timestep_delay)
        # return self.wave  # 2D array containing wave height data
        idx_wh11 = self.coordinates_to_indices((self.coords[0][0][0], self.coords[0][0][1]))
        idx_wh12 = self.coordinates_to_indices((self.coords[0][1][0], self.coords[0][1][1]))
        idx_wh21 = self.coordinates_to_indices((self.coords[1][0][0], self.coords[1][0][1]))
        idx_wh22 = self.coordinates_to_indices((self.coords[1][1][0], self.coords[1][1][1]))

        self.wave = np.array(
            [
                [self._wave_plane[idx_wh11], self._wave_plane[idx_wh12]],
                [self._wave_plane[idx_wh21], self._wave_plane[idx_wh22]],
            ]
        )
        #              |self.wave[-x,y]    self.wave[x,y] |
        #  self.wave = |                                  |
        #              |self.wave[-x,-y]   self.wave[x,-y]|
        return self.wave


class BuoyantBoat(gym.Env):
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, control_technique="DQN"):
        self.gravity = 10  # [m/s^2]
        # self.max_buyoancy = 35  # [N]
        self.max_action = 5
        self.dt = 0.1  # timestep [s]
        # self.crosssec_area = 3 * 8  # [m^2]
        # self.steady_sub_h = 1  # [m]
        self.density_water = 1000  # [kg/m^3]
        self.mass_boat = 11000  # [kg]
        # self.mass_boat = 60000  # [kg]
        # self.density_wood = 600  # [kg/m^3]
        self._dimensions = (4.0, 10.0, 2.0)
        self.relative_coordinates = np.array([0, 3, 2])
        self.width = self._dimensions[0]
        self.length = self._dimensions[1]
        self.height = self._dimensions[2]
        self.Ixx = (self.mass_boat / 12) * (
            self.height**2 + self.width**2
        )  # Moment of inertia about x-axis (kg m^2)
        self.Iyy = (self.mass_boat / 12) * (
            self.length**2 + self.width**2
        )  # Moment of inertia about y-axis (kg m^2)
        self.Izz = (self.mass_boat / 12) * (
            self.length**2 + self.height**2
        )  # Moment of inertia about z-axis (kg m^2)

        # state variables TODO: verify the cooridinate systems (base vs local), convert if necesary
        self.position = np.array([0, 0, -1], dtype=np.float64)  # [x, y, z] [m] Base Coordinate System
        self.prev_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]
        self.prev_velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]
        self.acceleration = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_accelerationn = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]
        self.prev_orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]
        self.angular_velocity = np.array([0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]
        self.prev_angular_velocity = np.array([0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]
        self.angular_acceleration = np.array([0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_angular_accelerationn = np.array([0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.combined_rotation_matrix = np.zeros(3)

        self._force_applied_coords = np.array(
            [  # top view, forward is up
                [
                    [self.length / 4, -self.width / 4, self.height / 2],  # top left
                    [self.length / 4, self.width / 4, self.height / 2],  # top right
                ],
                [
                    [-self.length / 4, -self.width / 4, self.height / 2],  # bottom left
                    [-self.length / 4, self.width / 4, self.height / 2],  # bottom right
                ],
            ],
            dtype=np.float64,
        )

        self.wave_generator = WaveGenerator(
            coords=np.array(
                [  # top view, forward is up
                    [
                        [self.length / 4, -self.width / 4, self.height / 2],  # top left
                        [self.length / 4, self.width / 4, self.height / 2],  # top right
                    ],
                    [
                        [-self.length / 4, -self.width / 4, self.height / 2],  # bottom left
                        [-self.length / 4, self.width / 4, self.height / 2],  # bottom right
                    ],
                ],
                dtype=np.float64,
            )
        )

        self.wave_state = self.wave_generator.update(0)
        self.step_count = 0
        self.winch_velocity = 0  # m/s

        self.rope_boat = 50  # Initial length of rope on boat side [Meters]
        self.rope_load = 10  # Initial length of rope on load side [Meters]
        self.wtb_dist = 15
        self.wtb_height = 50
        self.initial_rope_load = self.rope_load  # Initial length of rope on load side [Meters]

        # DQN spaces
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.rope_load,  # load position
            # self.wave_state[0][0]
        ]
        self.prev_state = copy.deepcopy(self.state)
        self.action = None

        self.forces = np.zeros((4, 3))  # 4 corners with [fx, fy, fz] [N]
        self.buoyant_forces = np.zeros((2, 2))
        self._torques = np.zeros((4, 3))
        self.total_torque = np.zeros(3)

        # PID controller parameters
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.prev_error = 0.0
        self.integral = 0.0

        # DQN parameters
        self.control_technique: Literal["DQN", "PID"] = control_technique

        self.csv_path = "./csv"
        self.rope_load_cache = []

    def reset(self, **kwargs):
        self.step_count = 0
        self.wave_state = self.wave_generator.update(self.step_count)

        self.position = np.array([0, 0, -1], dtype=np.float64)  # [x, y, z] [m] Base Coordinate System
        self.prev_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]

        self.velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]
        self.prev_velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]

        self.orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]
        self.prev_orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]

        self.angular_velocity = np.array([0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]
        self.prev_angular_velocity = np.array([0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]

        self.winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]

        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.rope_load,  # load position
            # self.wave_state[0][0]
        ]
        self.prev_state = copy.deepcopy(self.state)

        self.rope_load = self.initial_rope_load  # reset load side rope length to intial value
        self.integral = 0.0  # reset integral term for PID
        print("Reset")
        return self.state

    def rotation_x(self, roll):  # Rotation matrix for roll
        return np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    def rotation_y(self, pitch):  # Rotation matrix for pitch
        return np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    def rotation_z(self, yaw):  # Rotation matrix for yaw
        return np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    def get_buoyancy(self):
        # Calculate the combined rotation matrix
        self.combined_rotation_matrix = (
            self.rotation_z(self.orientation[2])
            @ self.rotation_y(self.orientation[1])
            @ self.rotation_x(self.orientation[0])
        )  # ZYX Tait-Bryan angles convention
        # Apply the rotation to the local coordinate system of the force application points
        # Then add the object's position to translate the points to the global coordinate system
        force_applied_points_global = (
            np.dot(self._force_applied_coords, self.combined_rotation_matrix.T) + self.position
        )
        # print(force_applied_points_global)
        force_applied_y = force_applied_points_global[:, :, 2:]
        self.buoyant_forces = np.zeros((2, 2))
        # Get max force so it doesn't go berserk on us later
        _max_force = self.density_water * self.gravity * self.width * self.height * self.length / 4
        # Calculate the buoyant forces acting on corner positions
        for i in range(2):
            for j in range(2):
                # local height = wave height - (current y position of force application - distance of force application to bottom deck)
                _hull_to_surface_distance = self.wave_state[i][j] - (
                    force_applied_y[i][j][0] - self._force_applied_coords[i][j][2]
                )
                _hull_area = self.width * self.length
                _local_buoyant_force = self.density_water * self.gravity * _hull_area / 4 * _hull_to_surface_distance

                if _local_buoyant_force > 0:
                    # print("we are underwater")
                    pass
                # cap forces between 0 and max, obviously
                self.buoyant_forces[i][j] = np.maximum(np.minimum(_local_buoyant_force, _max_force), 0)
        # print(self.buoyant_forces)
        if self.step_count == 6:
            # print("breakpoint")
            pass
        return self.buoyant_forces

    def apply_external_force(self, force):
        self.forces = force
        return np.sum(self.forces, axis=(0, 1))

    def calculate_torques(self):
        # torques = np.cross(self._force_applied_coords - self.position, self.forces)
        if self.step_count == 7:
            # print("breakpoint")
            pass
        torques = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                _dx = math.pow(self._force_applied_coords[:, :, :2][i][j][0], 2)
                _dy = math.pow(self._force_applied_coords[:, :, :2][i][j][1], 2)
                _distance_to_mc = math.pow(_dx + _dy, 0.5)
                torques[i][j] = _distance_to_mc * self.forces[i][j]  # sin = 1
        torque_pitch = (torques[0][1] + torques[1][1]) - (torques[0][0] + torques[1][0])  # (BL+BR) - (TL+TR)
        torque_roll = (torques[1][0] + torques[1][1]) - (torques[0][0] + torques[0][1])  # (BR+TR) - (BL+TL)
        return np.array((torque_pitch, torque_roll))

    def winch_control(self, action=0):
        if self.control_technique == "PID":  # check if PID is enabled
            _scale = 1
            _setpoint = 0
            error = _setpoint - self.rope_dy
            proportional_term = self.kp * error
            self.integral += error * self.dt
            integral_term = self.ki * self.integral
            derivative_term = self.kd * ((error - self.prev_error) / self.dt)
            pid_output = proportional_term + integral_term + derivative_term
            self.prev_error = error

            self.winch_velocity += _scale * pid_output
        elif self.control_technique == "DQN":
            self.winch_velocity = self.interpret_action(action)
        else:
            raise ValueError("No control technique specified")

        return self.winch_velocity

    def interpret_action(self, action):
        # print(f"Chosen action = {action}")
        if action == 0:
            self.winch_velocity = -5
        elif action == 1:
            self.winch_velocity = -4
        elif action == 2:
            self.winch_velocity = -3
        elif action == 3:
            self.winch_velocity = -2
        elif action == 4:
            self.winch_velocity = -1
        elif action == 5:
            self.winch_velocity = 0
        elif action == 6:
            self.winch_velocity = 1
        elif action == 7:
            self.winch_velocity = 2
        elif action == 8:
            self.winch_velocity = 3
        elif action == 9:
            self.winch_velocity = 4
        else:  # action == 10
            self.winch_velocity = 5
        return self.winch_velocity

    def DH_load(self):
        _prev_length = math.sqrt(
            math.pow(self.wtb_dist + self.prev_winch_position[1], 2)
            + math.pow(self.wtb_height - self.prev_winch_position[2], 2)
        )  # 15m from wtb, 50m height of wtb
        _curr_length = math.sqrt(
            math.pow(self.wtb_dist - self.winch_position[1], 2) + math.pow(self.wtb_height - self.winch_position[2], 2)
        )  # TODO: boat is not always facing forward, take that into account when calculating the distance
        self.rope_dy = _prev_length - _curr_length  # 15m from wtb, 50m height of wtb
        self.rope_boat -= self.rope_dy  # rope change on boat side
        self.winch_velocity = self.winch_control(self.action)  # TODO: winch model here
        self.rope_load += self.rope_dy + self.winch_velocity  # * self.dt  # rope change on load side
        return self.rope_boat, self.rope_load, self.winch_velocity

    def reward_function(self):
        rope_diff = abs(self.rope_load) - abs(
            self.initial_rope_load
        )  # Difference between rope on load side change with rope disturbance from boat
        scaling_factor = max(0, 1 - abs(rope_diff))  # Linear scaling
        reward = scaling_factor * 2 - 1  # Scale between -1 and 1 reward
        # reward = -abs(rope_diff)
        # print(reward)
        return reward

    def step(self, action):
        self.wave_state = self.wave_generator.update(self.step_count)

        self.action = action

        _current_buyoancy_forces = self.get_buoyancy()
        _total_external_forces = self.apply_external_force(_current_buyoancy_forces)
        total_torque = self.calculate_torques()

        # Linear motion equations
        sum_ext_forces = np.sum(_total_external_forces)
        # if sum_ext_forces > 1.5*self.mass_boat * self.gravity:
        #     sum_ext_forces = 1.5* self.mass_boat * self.gravity
        self.acceleration = (sum_ext_forces - self.mass_boat * self.gravity) / self.mass_boat
        self.velocity[2] += self.acceleration * self.dt
        self.position += self.velocity * self.dt

        # Angular motion equations for roll and pitch
        self.angular_acceleration = total_torque / np.array([self.Ixx, self.Iyy])
        # self.angular_acceleration = total_torque / np.array([self.Ixx, self.Iyy, self.Izz])
        self.angular_velocity += self.angular_acceleration * self.dt
        self.orientation[:2] += self.angular_velocity * self.dt
        # Clip the values in the array to the limits np.minimum(a_max, np.maximum(a, a_min))
        self.orientation = np.clip(self.orientation, -np.pi / 4, np.pi / 4)

        # TODO: DH model base coords based on rotation from self.orientation

        self.winch_position = np.dot(self.relative_coordinates, self.combined_rotation_matrix.T) + self.position

        _, self.rope_load, self.winch_velocity = self.DH_load()
        self.rope_load_cache.append(self.rope_load)

        # self.state[1] = self.rope_load
        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.rope_load,  # load position
            self.wave_state[0][0],
        ]

        reward = self.reward_function()
        self.step_count = self.step_count + 1
        if self.step_count > 400:
            done = True
            # self.write_to_csv(self.csv_path + "/rope_load.csv", self.rope_load_cache)
        else:
            done = False
        info = {}
        self.obs = np.array([self.position[2], self.rope_load])

        return (
            copy.deepcopy(self.state),
            _current_buyoancy_forces,
            done,
            False,
            info,
        )  # TODO: self.state should be self.obs

    def write_to_csv(self, path, data):
        with open(path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for data_point in data:
                csvwriter.writerow([data_point])


if __name__ == "__main__":
    env = BuoyantBoat()
    # env.reset()
    # env.step(0)
    _ = env.reset()

    state_positions = []
    state_orientations = []
    state_angular_velocities = []
    state_load_positions = []
    _current_buyoancy_forces = []
    wave_data = []

    for i in tqdm(range(10000)):
        # Take a step in the environment
        state, _current_buyoancy_force, _, _, _ = env.step(5)  # Assuming action 5 is used for all steps

        # Temporary variable to hold current state values
        state_positions.append(state[0])
        state_orientations.append(state[2])
        state_angular_velocities.append(state[3])
        state_load_positions.append(state[4])
        _current_buyoancy_forces.append(_current_buyoancy_force)
        wave_data.append(state[5])

    # Convert lists to numpy arrays for plotting
    np_positions = np.array(state_positions)
    np_orientations = np.array(state_orientations)
    np_angular_velocities = np.array(state_angular_velocities)
    np_load_position = np.array(state_load_positions)
    np_current_buyoancy_forces = np.array(_current_buyoancy_forces)
    np_wave_data = np.array(wave_data)

    # Plot states over time
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    # plt.plot(positions[:, 0], label='Position X')
    # plt.plot(positions[:, 1], label='Position Y')
    plt.plot(np_positions[:, 2], label="Position Z")
    plt.plot(np_wave_data, label="Wave position")
    plt.title("Position over time")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np_orientations[:, 0], label="Orientation Roll")
    plt.plot(np_orientations[:, 1], label="Orientation Pitch")
    # plt.plot(orientations[:, 2], label='Orientation Yaw')
    plt.title("Orientation over time")
    plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(np_angular_velocities[:, 0], label='Angular Velocity Roll')
    # plt.plot(np_angular_velocities[:, 1], label='Angular Velocity Pitch')
    # plt.plot(np_angular_velocities[:, 2], label='Angular Velocity Yaw')
    # plt.title('Angular Velocity over time')
    # plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(np_current_buyoancy_forces[:, 0, 0], label="buoyant force #1")
    plt.plot(np_current_buyoancy_forces[:, 1, 0], label="buoyant force #2")
    plt.plot(np_current_buyoancy_forces[:, 0, 1], label="buoyant force #3")
    plt.plot(np_current_buyoancy_forces[:, 1, 1], label="buoyant force #4")
    plt.title("buoyant forces")
    plt.legend()

    plt.tight_layout()
    plt.show()