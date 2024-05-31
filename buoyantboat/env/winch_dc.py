import matplotlib.pyplot as plt
import numpy as np

class WinchModelDC:
    def __init__(self, max_control_input):
        self.R = 1.0
        self.L = 0.5
        self.K = 0.1
        self.J = 0.02
        self.radius = 0.5  # radius of the winch
        self.I = 0.0  # Initial current
        self.omega = 0.0  # Initial angular velocity
        self.time = []
        self.velocity = []
        self.position = []
        self.velocity_lin=0
        self.max_speed = max_control_input

    def reset(self):
        self.I = 0.0  # Initial current
        self.omega=0.0
        self.velocity_lin=0.0
        self.time = []
        self.velocity = []
        self.position = []

    def step(self, up, dt):
        voltage = up
        self.I += ((voltage - self.I * self.R) / self.L) * dt
        torque = self.K * self.I
        alpha = torque / self.J
        self.omega += alpha * dt
        if abs(self.omega) > self.max_speed:
            if self.omega > 0.0:
                self.omega = self.max_speed
            elif self.omega < 0.0:
                self.omega = -self.max_speed
            
        self.time.append(dt * len(self.velocity))
        self.velocity.append(self.omega)
        self.velocity_lin += self.omega * self.radius

        return self.velocity_lin

    def plot_results(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.time, self.velocity)
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.title('Angular Velocity Over Time')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    motor_sim = WinchModelDC(max_control_input=3.0)
    dt = 0.01 
    voltage = 3.0

    for _ in range(10000):
        velocity = motor_sim.step(voltage, dt)
        if _ % 100 == 0:
            voltage = max(voltage - 0.1, 0.0)

    motor_sim.plot_results()