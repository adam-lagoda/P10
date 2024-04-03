import math
from typing import Literal
import csv
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WaveGenerator():
    def __init__(
            self,
            frequency_1=1,
            frequency_2=1,
            num_points=100,
            time_duration=1,
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
        self.x = np.linspace(-np.pi, np.pi, self.num_points)
        self.y = np.linspace(-np.pi, np.pi, self.num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.wave = self.wave1(self.X,0) + self.wave2(self.Y, 0, self.timestep_delay)

    def wave1(self, X, time):
        return 1/2 * np.sin(self.frequency_1 * (X + time * 2 * np.pi))

    def wave2(self, Y, time, delay):
        return 1/2 * np.sin(self.frequency_2 * (Y + (time - delay) * 2 * np.pi))

    def update(self, dt):
        self.wave = self.wave1(self.X, dt) + self.wave2(self.Y, dt, self.timestep_delay)
        return self.wave  # 2D array containing wave height data

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

        self.wave_state = 0
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
        self.state = np.array([0.0, 0.0])
        self.prev_state = np.array([0.0, 0.0])
        self.rope_load = self.initial_rope_load  # reset load side rope length to intial value
        self.wave_state = self.wave_generator[0]
        self.get_submersion()
        self.step_count = 0
        self.integral = 0.0  # reset integral term for PID
        print("Reset")
        return self.state

    def get_buyoancy(self, dy):
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

    # def reward_function(self):
    #     # Define your target position for the load
    #     target_position = self.initial_rope_load  # Replace this with the actual target position value

    #     # Calculate position error (how far the load is from the target position)
    #     position_error = self.rope_load - target_position
    #     position_error_penalty = -(position_error**2)

    #     # Calculate velocity of the load (change in position)
    #     load_velocity = self.state[0] - self.prev_state[0]  # Assuming self.state[0] represents load's position
    #     velocity_penalty = -(load_velocity**2)

    #     # Assign weights to the different components of the reward function
    #     # depending on which behavior you want to prioritize
    #     w_position = 1.0  # Weight for position error penalty
    #     w_velocity = 1.0  # Weight for velocity penalty

    #     reward = (w_position * position_error_penalty) + (w_velocity * velocity_penalty)

    #     # You might want to add an additional penalty for dropping the load into the water
    #     if self.rope_load < 0:
    #         reward -= 100  # Large negative penalty for dropping the load

    #     # Ensure the computation doesn't result in a positive reward
    #     reward = min(reward, 0.0)
    #     print(reward)
    #     return reward

    def step(self, action):
        self.wave_state = self.wave_generator[int(self.step_count) % len(self.wave_generator)]
        self.action = action

        dy = self.wave_state - self.state[0]
        self.prev_state[0] = self.boat_y
        _current_buyoancy_force = self.get_buyoancy(dy)
        self.net_force_boat = _current_buyoancy_force - self.mass_boat * self.gravity

        self.boat_y_ddot = self.net_force_boat / self.mass_boat
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
# 1. real wave data
#   1. ask if peter and daniel has wave data

# we optimize for disturbance, not time
