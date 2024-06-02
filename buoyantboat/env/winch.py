
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lsim, lti


class WinchModel:
    def __init__(self, max_control_input=1.0, mass_load=1000):
        # Define the system parameters
        self.m = mass_load
        self.k_oil = 1.8 * 10**9
        self.Vc = 2 * 10 ** (-3)
        self.Dp = 40 * 10 ** (-6)
        self.Dm = 4 * 10 ** (-6)
        self.w_p = 45
        self.Tp = 1.0
        self.k = 200
        self.r = 0.5
        self.eta_m = 0.65
        self.J_w = 150
        self.d = 10**5
        self.friction_coefficient = 10**3
        self.max_control_input = max_control_input

        # System matrices
        self.A = np.array(
            [
                [-1 / self.Tp, 0, 0, 0],
                [
                    -2 * self.k_oil * self.Dp * self.w_p / self.Vc,
                    0,
                    2 * (self.k_oil / self.Vc) * self.Dm * (self.k / self.r),
                    0,
                ],
                [
                    0,
                    -(self.r / (self.J_w + self.m * self.r**2))
                    * self.Dm
                    * self.k
                    * self.eta_m,
                    -(self.d + self.friction_coefficient)
                    / (self.J_w + self.m * self.r**2),
                    0,
                ],
                [0, 0, 1, 0],
            ]
        )
        self.B = np.array([[1 / self.Tp], [0], [0], [0]])
        self.C = np.array([[0, 0, 0, 1]])
        self.D = np.array([[0]])

        # LTI system representation
        self.sys = lti(self.A, self.B, self.C, self.D)

        self.reset()

    def reset(self):
        self.x0 = np.zeros(4)

    def get_winch_rotational_velocity(self, dt, target_velocity):
        target_velocity = np.clip(
            target_velocity, -self.max_control_input, self.max_control_input
        )
        t = np.array([0, dt])
        U = np.array([0, target_velocity])
        _, y, x = lsim(self.sys, U, t, self.x0)
        self.x0 = x[-1]  # Update the state vector for the next time step
        return y[-1]


if __name__ == "__main__":
    winch = WinchModel()

    # Simulate the response to a target velocity input
    dt = 0.01
    total_time = 50
    time_steps = int(total_time / dt)
    time = np.linspace(0, total_time, time_steps)
    target_velocity = 5.0  # target velocity = 5 rad/s
    response = []

    for t in time:
        if t > (total_time / 2):
            print("")
        velocity = winch.get_winch_rotational_velocity(dt, target_velocity)
        response.append(velocity)

    # with open('winch_target_velocity_response.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Time [s]', 'Winch Rotational Velocity'])
    #     writer.writerows(zip(time, response))

    # Plot the response
    plt.figure(figsize=(10, 6))
    plt.plot(time, response, label="Winch Rotational Velocity")
    plt.axhline(y=target_velocity, color="r", linestyle="--", label="Target Velocity")
    plt.title("Response of the Winch Model to Target Velocity Input")
    plt.xlabel("Time [s]")
    plt.ylabel("Winch Rotational Velocity")
    plt.grid(True)
    plt.legend()
    plt.show()
