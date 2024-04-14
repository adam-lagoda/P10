import math
from typing import Literal
import csv
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WaveGenerator():
    def __init__(
            self,
            frequency_1=0.25,
            frequency_2=0.25,
            num_points=100,
            time_duration=10,
            frames_per_second=24,
            timestep_delay=0.1):
        # Parameters
        self.frequency_1 = frequency_1  # frequency of 1st sine wave
        self.frequency_2 = frequency_2  # frequency of 2nd sine wave
        self.num_points = num_points  # number of points in each direction
        self.time_duration = time_duration  # duration of the animation in seconds
        self.frames_per_second = frames_per_second  # frames per second in the animation
        self.timestep_delay = timestep_delay  # delay in seconds for the second wave

        # Grid of points
        self.x = np.linspace(-10, 10, self.num_points)
        self.y = np.linspace(-10, 10, self.num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self._wave_plane = self.wave1(self.X,0) + self.wave2(self.Y, 0, self.timestep_delay)
        self.wave = None


    def _bilin_intp(self, x, y, arr=None):
        """Calculate a value based on non-int array index using bilinear interpolation."""
        if arr is None:
            arr = self._wave_plane
        # Get the integer parts of the indices
        x_floor = int(np.floor(x))
        x_ceil = int(np.ceil(x))
        y_floor = int(np.floor(y))
        y_ceil = int(np.ceil(y))

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
        x_weight = x - x_floor
        y_weight = y - y_floor

        # Perform bilinear interpolation
        top_interpolated = (top_right * x_weight) + (top_left * (1 - x_weight))
        bottom_interpolated = (bottom_right * x_weight) + (bottom_left * (1 - x_weight))
        interpolated_value = (bottom_interpolated * y_weight) + (top_interpolated * (1 - y_weight))

        return interpolated_value

    def wave1(self, X, time):
        return np.sin(self.frequency_1 * (X + time * 2 * np.pi))

    def wave2(self, Y, time, delay):
        return np.sin(self.frequency_2 * (Y + (time - delay) * 2 * np.pi))

    def update(self, dt):
        self._wave_plane = self.wave1(self.X, dt) + self.wave2(self.Y, dt, self.timestep_delay)
        # return self.wave  # 2D array containing wave height data
        _mc = (50, 50)
        idx_wh11 = [_mc[0]-5, _mc[1]+12.5]
        idx_wh12 = [_mc[0]+5, _mc[1]+12.5]
        idx_wh21 = [_mc[0]-5, _mc[1]-12.5]
        idx_wh22 = [_mc[0]+5, _mc[1]-12.5]

        self.wave = np.array(
            [
                [
                    self._bilin_intp(idx_wh11[0], idx_wh11[1]), self._bilin_intp(idx_wh12[0], idx_wh12[1])
                ],
                [
                    self._bilin_intp(idx_wh21[0], idx_wh21[1]), self._bilin_intp(idx_wh22[0], idx_wh22[1])
                ]
            ]
        )
        #              |self.wave[-x,y]    self.wave[x,y] |
        #  self.wave = |                                  |
        #              |self.wave[-x,-y]   self.wave[x,-y]|
        return self.wave

class BuoyantBoat(gym.Env):
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, control_technique="DQN"):
        self.gravity = 10  # [m/s^2]
        self.max_buyoancy = 35  # [N]
        self.max_action = 5
        self.dt = 0.01  # timestep [s]
        self.crosssec_area = 3 * 8  # [m^2]
        self.steady_sub_h = 1  # [m]
        self.density_water = 1000  # [kg/m^3]
        self.mass_boat = 50  # [kg]
        self.density_wood = 600  # [kg/m^3]

        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2, ))
        self.state = np.zeros(2)  # y axis of the boat, y axis of the load
        self.action = None

        self.wave_generator = WaveGenerator()
        self.wave_state = self.wave_generator.update(0)
        self.step_count = 0
        self.net_force_boat = 0
        self.boat_y_ddot = 0
        self.boat_y_dot = 0
        self.boat_y = 0
        self.prev_state = np.zeros(1)
        self.winch_velocity = 0  # m/s

        self.rope_boat = 50  # Initial length of rope on boat side [Meters]
        self.rope_load = 10  # Initial length of rope on load side [Meters]
        self.initial_rope_load = self.rope_load  # Initial length of rope on load side [Meters]

        self.wtb_dist = 15
        self.wtb_height = 50

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
        self.state = np.array([np.array([[0.0, 0.0], [0.0, 0.0]]), 0.0])  # boat state, load state
        self.prev_state = np.array([0.0, 0.0])
        self.rope_load = self.initial_rope_load  # reset load side rope length to intial value
        self.get_submersion()
        self.step_count = 0
        self.wave_state = self.wave_generator.update(self.step_count)
        self.integral = 0.0  # reset integral term for PID
        print("Reset")
        return self.state

    def get_buyoancy(self, dy):
        # return self.density_water * self.crosssec_area * (self.steady_sub_h + dy) * self.gravity  # [N]
        
        return self.density_water * self.crosssec_area * (self.steady_sub_h + dy) * self.gravity  # [N]

    def get_submersion(self):
        """Inverse of func::self.get_buyoancy() for a case when buyoancy force is equal to mass*gravity."""
        self.steady_sub_h = self.mass_boat / (self.density_water * self.crosssec_area)  # [N]

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
        print(f"Chosen action = {action}")
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

    def rope_length(self):
        _prev_length = math.sqrt(
            math.pow(self.wtb_dist, 2) + math.pow(self.wtb_height - self.prev_state[0], 2)
        )  # 15m from wtb, 50m height of wtb
        _curr_length = math.sqrt(math.pow(self.wtb_dist, 2) + math.pow(self.wtb_height - self.state[0], 2))
        self.rope_dy = _prev_length - _curr_length  # 15m from wtb, 50m height of wtb
        self.rope_boat -= self.rope_dy  # rope change on boat side
        self.winch_velocity = self.winch_control(self.action)
        self.rope_load += self.rope_dy + self.winch_velocity #* self.dt  # rope change on load side
        return self.rope_boat, self.rope_load, self.winch_velocity

    def reward_function(self):
        rope_diff = abs(self.rope_load) - abs(self.initial_rope_load)  # Difference between rope on load side change with rope disturbance from boat
        scaling_factor = max(0, 1 - abs(rope_diff))  # Linear scaling
        reward = scaling_factor * 2 - 1  # Scale between -1 and 1 reward
        # reward = -abs(rope_diff)
        print(reward)
        return reward

    def step(self, action):
        self.wave_state = self.wave_generator.update(self.step_count)

        self.action = action

        dy = self.wave_state - self.state[0]  # TODO: all of the concurrent operations need to be done element-wise
        self.prev_state[0] = self.boat_y
        _current_buyoancy_force = self.get_buyoancy(dy)
        self.net_force_boat = _current_buyoancy_force - self.mass_boat * self.gravity

        self.boat_y_ddot = self.net_force_boat / self.mass_boat  # TODO: rotation matrix needs to be implemented based on force applied
        self.boat_y_dot = self.boat_y_ddot * self.dt  # *dt
        self.boat_y = self.boat_y_dot * self.dt  # *dt

        self.state[0] = self.boat_y

        _, self.rope_load, self.winch_velocity = self.rope_length()
        self.rope_load_cache.append(self.rope_load)

        self.state[1] = self.rope_load

        reward = self.reward_function()
        self.step_count = self.step_count + 1
        if self.step_count > 400:
            done = True
            self.write_to_csv(self.csv_path + "/rope_load.csv", self.rope_load_cache)
        # elif self.rope_load < 0:
        #     done = True
        else:
            done = False
        info = {}
        self.obs = np.array([self.state[0], self.state[1]])

        if self.control_technique == "DQN":
            return self.obs, reward, done, False, info
        else:
            return (
                self.boat_y_ddot,
                self.boat_y_dot,
                self.boat_y,
                self.wave_state,
                self.rope_load,
                self.winch_velocity,
                0.0,
                False,
                {},
            )

    def write_to_csv(self, path, data):
        with open(path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for data_point in data:
                csvwriter.writerow([data_point])

## TODO list:
# we optimize for disturbance, not time