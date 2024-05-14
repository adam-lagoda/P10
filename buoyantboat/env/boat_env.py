import copy
import csv
import math
from typing import Literal

import gymnasium as gym
import numpy as np
from buoyantboat.env.dh_transform import calculate_dh_rotation_matrice
from buoyantboat.env.wave_generator import WaveGenerator
from buoyantboat.env.winch import WinchModel
from gymnasium import spaces
from tqdm import tqdm
import random


class BuoyantBoat(gym.Env):
    def __init__(
            self,
            control_technique="DQN",
            target_position=None,
            target_velocity=None,
            max_step_per_episode=1000
        ):
        self.gravity = 10  # [m/s^2]
        # self.max_buyoancy = 35  # [N]
        self.max_action = 5
        self.dt = 0.05  # timestep [s]
        # self.crosssec_area = 3 * 8  # [m^2]
        # self.steady_sub_h = 1  # [m]
        self.density_water = 1000  # [kg/m^3]
        self.mass_boat = 4000  # [kg]
        # self.mass_boat = 10000  # [kg]
        self.mass_load = 1000  # [kg]
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
        self.position = np.array([0, 0, -0.34], dtype=np.float64)  # [x, y, z] [m] Base Coordinate System
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
        # self.prev_winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
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
        self.max_step_per_episode = max_step_per_episode
        max_winch_speed = 3.0
        self.winch = WinchModel(
            max_control_input=max_winch_speed,
            mass_load=self.mass_load
        )
        self.winch.reset()
        self.winch_velocity = 0  # m/s

        # self.rope_length_boat_side = 50  # length of rope on boat side [Meters]
        # self.rope_length_load_side = 30  # length of rope on load side [Meters]
        self.wtb_dist = 15
        self.wtb_height = 50
        # self.initial_rope_length_boat_side = copy.deepcopy(self.rope_length_boat_side)  # Initial length of rope on boat side [Meters]
        # self.initial_rope_length_load_side = copy.deepcopy(self.rope_length_load_side)  # Initial length of rope on load side [Meters]
        # self.rope_dy = 0
        self.rope_length = 75
        self.wtb_crane_position = np.array([self.wtb_dist, 0, self.wtb_height], dtype=np.float64)  # [x, y, z] [m]
        self.load_position = np.array(
            [
                self.wtb_crane_position[0],
                self.wtb_crane_position[1],
                self.wtb_crane_position[2] - 24,
            ],
            dtype=np.float64,
        )
        self.prev_load_postion = copy.deepcopy(self.load_position)
        self.load_velocity_z = 0

        # DQN spaces
        if control_technique == "SAC":
            self.action_space = spaces.Box(low=-max_winch_speed, high=max_winch_speed, shape=(1,), dtype=np.float64) # SAC - continuous action_space
        else:
            self.action_space = spaces.Discrete(11)  # DQN - discrete action_space

        self.observation_space = spaces.Box(low=-5, high=50, shape=(6,))
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.load_position,  # load position
            self.wave_state[0][0],  # wave heave in top left corner for debugging
            None,  # current buoyant forces
        ]
        self.prev_state = copy.deepcopy(self.state)
        self.action = None

        self.forces = np.zeros((4, 3))  # 4 corners with [fx, fy, fz] [N]
        self.buoyant_forces = np.zeros((2, 2))
        self._torques = np.zeros((4, 3))
        self.total_torque = np.zeros(3)

        # DQN parameters
        self.control_technique: Literal["DQN", "SAC"] = control_technique

        self.csv_path = "./csv"
        self.rope_load_cache = []

        self.target_velocity = target_velocity
        self.target_position = target_position
        self.updated_already = False

    def reset(self, **kwargs):
        super().reset()
        self.step_count = 0
        self.wave_generator = WaveGenerator(
            coords=self._force_applied_coords,
            amplitude_1=random.uniform(0.15, 0.35),
            amplitude_2=random.uniform(0.15, 0.35)
            )

        self.wave_state = self.wave_generator.update(self.step_count)

        self.position = np.array([0, 0, -0.34], dtype=np.float64)  # [x, y, z] [m] Base Coordinate System
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
        self.winch.reset()
        self.winch_velocity = 0  # m/s
        # self.prev_winch_position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]

        self.load_position = np.array(
            [
                self.wtb_crane_position[0],
                self.wtb_crane_position[1],
                # self.wtb_crane_position[2] - 24,
                self.wtb_crane_position[2] - 5,
            ],
            dtype=np.float64,
        )
        self.prev_load_postion = copy.deepcopy(self.load_position)
        self.load_velocity_z = 0

        self.added_rope_winch = 0
        self.rope_length = 75
        
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

        if self.target_position is None or self.updated_already is True:
            self.target_position = random.randint(10, 25)
        if self.target_velocity is None or self.updated_already is True:
            # self.target_velocity = random.uniform(-1.5, -0.2)
            self.target_velocity = 0
        self.updated_already=True
            # self.target_velocity = 0
        print(f"This episode's targets: p={self.target_position}, v={self.target_velocity}.")

        obs = np.array(  # same as self.observation_space
            [
                self.load_position[2],  # load position
                self.target_position,
                self.load_velocity_z,
                self.target_velocity,
                self.position[2],  # boat position
                self.velocity[2]  # boat velocity
            ],
            dtype=np.float64,
        )
        print("Reset")
        info = {}
        return obs, info

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
        winch_velocity = self.interpret_action(action)

        return winch_velocity

    def interpret_action(self, action):
        # print(f"Chosen action = {action}")
        if self.control_technique == "SAC":
            self.winch_velocity = action[0]
            # winch_velocity = self.winch.get_winch_rotational_velocity(
            #     dt=self.dt,
            #     num_steps=self.step_count,
            #     up=action[0]
            # )
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


        self.winch_velocity = self.winch_control(action)
        self.rope_length += self.winch_velocity * self.dt

        # UNCOMMENT BEFORE SUBMISSION
        # total_displacement_matrix = calculate_dh_rotation_matrice(
        #     _theta=np.pi/2 - angle,
        #     initial_boat_rope_length=full_magnitude,
        #     intital_load_rope_length=self.rope_length-full_magnitude,
        # )
        # p_vec = np.array([self.winch_position[0], self.winch_position[1], self.winch_position[2], 1])
        # # TODO: is the order of coordinates correct
        # # self.load_position = np.dot(p_vec, total_displacement_matrix.T)
        # position = total_displacement_matrix @ p_vec
        # self.load_position = np.array([position[2], position[1], position[0]])
        
        # HACKY SOLUTION
        
        self.load_position = np.array(
            [
                self.wtb_crane_position[0],
                self.wtb_crane_position[1],
                self.wtb_crane_position[2]-(self.rope_length-full_magnitude)
            ]
        )
        
        return self.load_position, self.winch_velocity


    # def reward_function(self, target_velocity, target_position):
    #     # old
    #     # reward = -10*abs(target_velocity-self.load_velocity_z)
    #     # new
    #     # reward_velocity = -abs(target_velocity-self.load_velocity_z) + 1
    #     done = False
    #     velocity_error = abs(self.load_velocity_z-target_velocity)
    #     max_error_threshold = 1.0
    #     if velocity_error <= max_error_threshold:
    #         reward_velocity = 1 - (velocity_error/max_error_threshold)
    #     else:
    #         reward_velocity = -((velocity_error-max_error_threshold)/max_error_threshold)
    #     # reward_velocity = -abs(target_velocity-self.load_velocity_z)+1
    #     weight_velocity = 10.0

    #     position_error = target_position-self.load_position[2]
    #     reward_position = -abs(position_error)+1
    #     weight_position = 10.0
    #     reward = weight_position * reward_position + weight_velocity * reward_velocity

    #     # Sanity checks
    #     # if self.step_count > self.max_step_per_episode:
    #         # if abs(self.load_velocity_z) > 0.5+abs(target_velocity):
    #         #     reward -= 20
    #         #     done = True
    #     if self.load_position[2] < target_position+1 and self.load_position[2] > target_position-1:
    #             # if self.load_velocity_z > -0.1 and self.load_velocity_z < 0.1:
    #         reward += 20
    #             # done = True
    #     # print(f"Reward={reward}")
    #     return reward, done


    def reward_function(self, target_velocity, target_position):
        done = False

        # Velocity error component
        velocity_error = abs(target_velocity - self.load_velocity_z)
        max_velocity_error = 1.0  # Define a maximum tolerable velocity error
        
        # Reward for velocity should be a negative quadratic function of error, 
        # which means the penalty increases as the error gets larger.
        reward_velocity = -velocity_error**2

        # Position error component
        position_error = abs(target_position - self.load_position[2])
        reward_position = -position_error**2  # Use a negative quadratic function for position error as well
        
        # Weighting factors for velocity and position rewards
        weight_velocity = 1.0
        weight_position = 1.0

        # Combine weighted rewards
        reward = (weight_position * reward_position) + (weight_velocity * reward_velocity)
        # reward = reward_position

        # Check for termination
        if self.step_count > 100:
            # Check if the position is within a desirable threshold from the target
            position_threshold = 0.5  # This defines a band around the target position that is considered acceptable
            if abs(position_error) < position_threshold:
                # Provide a bonus for maintaining position within the threshold
                reward = reward + 20
                # done = True
        return reward, done

    def _get_obs(self, action):
        self.wave_state = self.wave_generator.update(self.step_count)

        self.action = action
        # print(f"Action={action}")

        # Linear motion equations
        _current_buyoancy_forces = self.get_buoyancy()
        _total_external_forces = self.apply_external_force(_current_buyoancy_forces)
        # _drag_coefficient = 10000.5
        # _drag_coefficient = 0.5  # cube
        _drag_coefficient = 2.0  # cube
        if _total_external_forces > 0.0:  # buoyant forces working <-> we are underwater <-> viscous friction is acting
            _viscous_friction = -_drag_coefficient * self.width * self.length * self.density_water * np.square(self.velocity[2])/2  # TODO: link to paper
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
        self.load_velocity_z = (self.load_position[2]-self.prev_load_postion[2])/self.dt
        # print(f"winch_dl={self.winch_velocity*self.dt}")
        # self.rope_load_cache.append(self.rope_length_load_side)
        # self.prev_winch_position = copy.deepcopy(self.winch_position)

        # self.state[1] = self.rope_load
        self.state = [
            self.position,  # boat position
            self.velocity,  # boat velocity
            self.orientation,  # boat orientation
            self.angular_velocity,  # boat angular velocity
            self.load_position,  # load position
            self.wave_state[0][0],  # wave heave in top left corner for debugging
            self.velocity[2],  # boat velocity
            _current_buyoancy_forces,  # current buoyant forces
        ]

        obs = np.array(  # same as self.observation_space
            [
                self.load_position[2],  # load position
                self.target_position,
                self.load_velocity_z,
                self.target_velocity,
                self.position[2],  # boat position
                self.velocity[2]  # boat velocity
            ],
            dtype=np.float64,
        )
        
        self.prev_load_postion = copy.deepcopy(self.load_position)
        
        return obs

    def step(self, action):
        obs = self._get_obs(action)
        reward, done = self.reward_function(
            target_position=self.target_position,
            target_velocity=self.target_velocity
        )
        self.step_count = self.step_count + 1
        if done is False:
            if self.step_count > self.max_step_per_episode:
                done = True
                # self.write_to_csv(self.csv_path + "/rope_load.csv", self.rope_load_cache)
            else:
                done = False

        info ={
            "load_position": self.load_position[2],  # load position
            "target_position": self.target_position,
            "load_velocity": self.load_velocity_z,
            "target_velocity": self.target_velocity,
            "winch_velocity": self.winch_velocity,
            # "position": self.position[2],  # boat position
            # "velocity": self.velocity[2]  # boat velocity
        }
        return (
            obs,
            # copy.deepcopy(self.state),  # COMMENT OUT FOR TRAINING
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