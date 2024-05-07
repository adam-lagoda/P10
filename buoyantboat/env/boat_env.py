import copy
import csv
import math
from typing import Literal

import gymnasium as gym
import numpy as np
from buoyantboat.env.dh_transform import calculate_dh_rotation_matrice
from gymnasium import spaces
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
        self.dt = 0.05  # timestep [s]
        # self.crosssec_area = 3 * 8  # [m^2]
        # self.steady_sub_h = 1  # [m]
        self.density_water = 1000  # [kg/m^3]
        self.mass_boat = 6000  # [kg]
        # self.mass_boat = 60000  # [kg]
        # self.density_wood = 600  # [kg/m^3]
        self._dimensions = (2.0, 5.0, 1.0)
        self.relative_coordinates = np.array([3, 0, 2], dtype=np.float64)
        self.width = self._dimensions[0]
        self.length = self._dimensions[1]
        self.height = self._dimensions[2]
        self.Ixx = (self.mass_boat / 12) * (self.height**2 + self.width**2)  # Moment of inertia about x-axis (kg m^2)
        self.Iyy = (self.mass_boat / 12) * (self.length**2 + self.width**2)  # Moment of inertia about y-axis (kg m^2)
        self.Izz = (self.mass_boat / 12) * (self.length**2 + self.height**2)  # Moment of inertia about z-axis (kg m^2)

        # state variables TODO: verify the cooridinate systems (base vs local), convert if necesary
        self.position = np.array([0, 0, -1], dtype=np.float64)  # [x, y, z] [m] Base Coordinate System
        self.prev_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]
        self.prev_velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]
        self.acceleration = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_accelerationn = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]
        self.prev_orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]
        self.angular_velocity = np.array([0, 0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]
        self.prev_angular_velocity = np.array([0, 0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]
        self.angular_acceleration = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_angular_accelerationn = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
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

        self.wave_generator = WaveGenerator(coords=self._force_applied_coords)

        self.wave_state = self.wave_generator.update(0)
        self.step_count = 0
        self.winch_velocity = 0  # m/s

        self.rope_length_boat_side = 50  # length of rope on boat side [Meters]
        self.rope_length_load_side = 30  # length of rope on load side [Meters]
        self.wtb_dist = 15
        self.wtb_height = 50
        self.initial_rope_length_boat_side = copy.deepcopy(self.rope_length_boat_side)  # Initial length of rope on boat side [Meters]
        self.initial_rope_length_load_side = copy.deepcopy(self.rope_length_load_side)  # Initial length of rope on load side [Meters]
        self.rope_dy = 0
        self.added_rope_winch = 0
        self.wtb_crane_position = np.array([self.wtb_dist, 0, self.wtb_height], dtype=np.float64)  # [x, y, z] [m]
        self.load_position = np.array(
            [
                self.wtb_crane_position[0],
                self.wtb_crane_position[1],
                self.wtb_crane_position[2] - self.initial_rope_length_load_side,
            ],
            dtype=np.float64,
        )
        self.prev_load_postion = copy.deepcopy(self.load_position)

        # DQN spaces
        if control_technique == "SAC":
            self.action_space = spaces.Box(low=-5, high=5, shape=(1,), dtype=np.float64)
        else:
            self.action_space = spaces.Discrete(11)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.rope_length_load_side,  # load position
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
        self.control_technique: Literal["DQN", "PID", "SAC"] = control_technique

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

        self.angular_velocity = np.array([0, 0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]
        self.prev_angular_velocity = np.array(
            [0, 0, 0], dtype=np.float64
        )  # [roll_rate, pitch_rate, yaw_rate] [rad/s]

        self.winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.prev_winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]

        self.load_position = np.array(
            [
                self.wtb_crane_position[0],
                self.wtb_crane_position[1],
                self.wtb_crane_position[2] - self.initial_rope_length_load_side,
            ],
            dtype=np.float64,
        )
        self.prev_load_postion = copy.deepcopy(self.load_position)

        self.added_rope_winch = 0
        
        self.state = np.array(
            [
                self.position,  # boat position
                self.velocity,  # boat velocity
                self.orientation,  # boat orientation
                self.angular_velocity,  # boat angular velocity
                self.load_position,  # load position
                # self.wave_state[0][0]
            ],
            dtype=np.float64,
        )
        self.prev_state = copy.deepcopy(self.state)

        self.obs = np.array(
            [
                self.position[2],  # boat position
                # self.velocity[2],  # boat velocity
                # self.orientation[0],  # boat orientation
                # self.orientation[1],  # boat orientation
                # self.angular_velocity,  # boat angular velocity
                self.load_position[2],  # load position
                # self.winch_velocity
            ],
            dtype=np.float64,
        )

        self.rope_length_load_side = self.initial_rope_length_load_side  # reset load side rope length to intial value
        self.rope_length_boat_side = self.initial_rope_length_boat_side  # reset boat side rope length to intial value
        self.integral = 0.0  # reset integral term for PID
        print("Reset")
        info = {}
        return self.obs, info

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
        plane_normal_local = np.dot(
            self.combined_rotation_matrix, np.array([0, 0, 1])
        )  # Global combined rotation matrix dot z axis unit vector
        plane_normal_unit = plane_normal_local / np.linalg.norm(plane_normal_local)
        dot_product = np.dot(plane_normal_unit, np.array([0, 0, 1]))
        for i in range(2):
            for j in range(2):
                _dx = math.pow(self._force_applied_coords[:, :, :2][i][j][0], 2)
                _dy = math.pow(self._force_applied_coords[:, :, :2][i][j][1], 2)
                _distance_to_mc = math.pow(_dx + _dy, 0.5)
                torques[i][j] = (
                    _distance_to_mc * self.forces[i][j] * np.sin(np.arcsin(np.clip(dot_product, -1, 1)))
                )  # TODO: add sine of plane normal angle
        torque_pitch = (torques[0][1] + torques[1][1]) - (torques[0][0] + torques[1][0])  # (BL+BR) - (TL+TR)
        torque_roll = (torques[1][0] + torques[1][1]) - (torques[0][0] + torques[0][1])  # (BR+TR) - (BL+TL)
        return np.array((torque_pitch, torque_roll))

    def winch_control(self, action):
        self.winch_velocity = self.interpret_action(action)

        return self.winch_velocity

    def interpret_action(self, action):
        # print(f"Chosen action = {action}")
        if self.control_technique == "SAC":
            self.winch_velocity = action[0]
            return self.winch_velocity
        else:
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


    def compute_dh_model(self, action):
        # Calculate the vector between the winch and the crane
        vector_x = self.wtb_crane_position[0] - self.winch_position[0]
        vector_y = self.wtb_crane_position[1] - self.winch_position[1]
        vector_z = self.wtb_crane_position[2] - self.winch_position[2]
        # Calculate the magnitude (length) of the vector in the XY plane
        xy_magnitude = math.sqrt(vector_x**2 + vector_y**2)
        # Calculate the magnitude (length) of the full vector
        full_magnitude = math.sqrt(vector_x**2 + vector_y**2 + vector_z**2)
        angle = math.acos(xy_magnitude / full_magnitude)  # TODO: what is the reference to which we calculate angles

        # Calculate the distance between the 2 points (winch and crane) in 3D
        _squared_differences = np.square(self.winch_position - self.wtb_crane_position)
        _sum_of_squares = np.sum(_squared_differences)
        _curr_length = np.sqrt(_sum_of_squares)
        if self.step_count == 0:  # initial condition
            _prev_length = _curr_length
            self.initial_rope_length_boat_side = full_magnitude
        else:
            _prev_squared_differences = np.square(self.prev_winch_position - self.wtb_crane_position)
            _prev_sum_of_squares = np.sum(_prev_squared_differences)
            _prev_length = np.sqrt(_prev_sum_of_squares)

        self.rope_dy = _prev_length - _curr_length
        print(f"rope_dy={self.rope_dy}")

        self.winch_velocity = self.winch_control(action)
        self.added_rope_winch += self.winch_velocity * self.dt

        total_displacement_matrix = calculate_dh_rotation_matrice(
            _theta=np.pi/2 - angle,
            d1=-self.rope_dy,
            d2=self.rope_dy + self.added_rope_winch,
            initial_boat_rope_length=self.initial_rope_length_boat_side,
            intital_load_rope_length=self.initial_rope_length_load_side,
        )
        p_vec = np.array([self.winch_position[0], self.winch_position[1], self.winch_position[2], 1])
        # TODO: is the order of coordinates correct
        # self.load_position = np.dot(p_vec, total_displacement_matrix.T)
        position = total_displacement_matrix @ p_vec
        self.load_position = np.array([position[2], position[1], position[0]])
        return self.load_position, self.winch_velocity


    def reward_function(self, preset=0):
        # scaling_factor = max(0, 1 - abs(self.rope_dy))  # Linear scaling
        # reward = scaling_factor * 2 - 1  # Scale between -1 and 1 reward
        # reward = -10*abs(self.rope_dy)
        load_velocity = (self.prev_load_postion[2] - self.load_position[2])/self.dt
        reward = -10*abs(preset-load_velocity)
        # print(f"Reward={reward}")
        return reward

    def step(self, action):
        self.wave_state = self.wave_generator.update(self.step_count)

        self.action = action
        print(f"Action={action}")

        # Linear motion equations
        _current_buyoancy_forces = self.get_buoyancy()
        _total_external_forces = self.apply_external_force(_current_buyoancy_forces)
        _drag_coefficient = 0.5
        if _total_external_forces > 0.0:  # buoyant forces working <-> we are underwater <-> viscous friction is acting
            _viscous_friction = -_drag_coefficient * np.abs(self.velocity[2])  # TODO: revisit and verify
        else:
            _viscous_friction = 0.0
        sum_ext_forces = np.sum(_total_external_forces) + _viscous_friction
        _total_forces = sum_ext_forces - self.mass_boat * self.gravity
        self.acceleration = _total_forces / self.mass_boat
        self.velocity[2] += self.acceleration * self.dt
        self.position += self.velocity * self.dt

        # Angular motion equations for roll and pitch
        total_torque = self.calculate_torques()
        self.angular_acceleration = total_torque / np.array([self.Ixx, self.Iyy])
        d_angular_velocity_x = self.angular_acceleration[0] * self.dt
        d_angular_velocity_y = self.angular_acceleration[1] * self.dt
        self.angular_velocity += np.array(
            [d_angular_velocity_x, d_angular_velocity_y, 0],
            dtype=np.float64,
        )
        self.orientation += self.angular_velocity * self.dt
        # self.orientation = np.clip(self.orientation, -np.pi / 4, np.pi / 4)

        # Get position of the winch in Global Coordinate System based on position
        # relative to center of mass and a combined rotation matrix
        self.winch_position = np.dot(self.relative_coordinates, self.combined_rotation_matrix.T) + self.position

        self.load_position, self.winch_velocity = self.compute_dh_model(action)
        print(f"winch_dl={self.winch_velocity*self.dt}")
        # self.rope_load_cache.append(self.rope_length_load_side)
        self.prev_winch_position = copy.deepcopy(self.winch_position)
        self.prev_load_postion = copy.deepcopy(self.load_position)

        # self.state[1] = self.rope_load
        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.load_position,  # load position
            self.wave_state[0][0],  # wave heave in top left corner for debugging
            _current_buyoancy_forces,  # current buoyant forces
        ]

        self.obs = np.array(  # TODO: change to self.observation_space
            [
                self.position[2],  # boat position
                self.load_position[2],  # load position
            ],
            dtype=np.float64,
        )

        reward = self.reward_function()
        self.step_count = self.step_count + 1
        if self.step_count > 400:
            done = True
            # self.write_to_csv(self.csv_path + "/rope_load.csv", self.rope_load_cache)
        else:
            done = False

        info = {}

        return (
            # copy.deepcopy(self.obs),
            copy.deepcopy(self.state),  # COMMENT OUT FOR TRAINING
            reward,
            done,
            False,
            info,
        )

    def write_to_csv(self, path, data):
        with open(path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            for data_point in data:
                csvwriter.writerow([data_point])