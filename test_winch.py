import numpy as np
import matplotlib.pyplot as plt

class WinchModel:
    def __init__(self, max_control_input=1.0, mass_load=1000):
        # Define the system parameters
        self.m = mass_load
        self.k_oil = 1.8 * 10 ** 9
        self.Vc = 2 * 10 ** (-3)
        self.Dp = 40 * 10 ** (-6)
        self.Dm = 4 * 10 ** (-6)
        self.w_p = 45
        self.Tp = 1.0
        self.k = 200
        self.r = 0.5
        self.eta_m = 0.65
        self.J_w = 150
        self.d = 10 ** 4  # Reduced damping coefficient for stability
        self.friction_coefficient = 10 ** 3
        self.max_control_input = max_control_input

        # Initialize the state vector
        self.x0 = np.zeros(4)

    def winch_dynamics(self, x, u):
        dx = np.zeros_like(x)
        dx[0] = -1 / self.Tp * x[0]
        dx[1] = -2 * self.k_oil * self.Dp * self.w_p / self.Vc * x[1] + 2 * (self.k_oil / self.Vc) * self.Dm * (self.k / self.r) * u
        dx[2] = -(self.r / (self.J_w + self.m * self.r ** 2)) * self.Dm * self.k * self.eta_m * x[1] - (self.d + self.friction_coefficient) / (self.J_w + self.m * self.r ** 2) * x[2]
        dx[3] = x[2]
        return dx

    def simulate(self, dt, total_time, target_velocity):
        num_steps = int(total_time / dt)
        time = np.linspace(0, total_time, num_steps + 1)
        velocities = np.zeros(num_steps + 1)
        x = self.x0.copy()

        for i in range(num_steps):
            u = np.clip(target_velocity - x[3], -self.max_control_input, self.max_control_input)
            k1 = self.winch_dynamics(x, u)
            k2 = self.winch_dynamics(x + 0.5 * dt * k1, u)
            k3 = self.winch_dynamics(x + 0.5 * dt * k2, u)
            k4 = self.winch_dynamics(x + dt * k3, u)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            velocities[i + 1] = x[3]

        return time, velocities

# Create an instance of WinchModel
winch = WinchModel()

# Simulation parameters
dt = 0.001  # Reduced integration step size
total_time = 50
target_velocity = 5.0  # Target velocity

# Simulate the response
time, response = winch.simulate(dt, total_time, target_velocity)

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(time, response, label='Winch Rotational Velocity')
plt.axhline(y=target_velocity, color='r', linestyle='--', label='Target Velocity')
plt.title('Response of the Winch Model to Target Velocity Input')
plt.xlabel('Time [s]')
plt.ylabel('Winch Rotational Velocity')
plt.grid(True)
plt.legend()
plt.show()
