import numpy as np
from scipy.integrate import solve_ivp

class WinchModel:
    def __init__(self, max_control_input=1.0, mass_load=1000):
        # Define the system parameters
        self.m = mass_load
        self.k_oil = 1.8*10**9
        self.Vc = 2*10**(-3)
        self.Dp = 40*10**(-6)
        self.Dm = 4*10**(-6)
        self.w_p = 45
        self.Tp = 1.0
        self.k = 200
        self.r = 0.5
        self.eta_m = 0.65
        self.J_w = 150
        # self.J_w = 0.1
        self.d = 10**4
        self.max_control_input = max_control_input
    
        # Define the system matrices
        self.A_11 = -1 / self.Tp
        self.A_21 = -2 * self.k_oil * self.Dp * self.w_p / self.Vc
        self.A_23 = 2 * (self.k_oil / self.Vc) * self.Dm * (self.k / self.r)
        self.A_32 = -(self.r / (self.J_w + self.m * self.r ** 2)) * self.Dm * self.k * self.eta_m
        self.A_33 = -self.d / (self.J_w + self.m * self.r ** 2)
        self.A_43 = 1
        self.B_11 = 1 / self.Tp

        self.reset()

    def reset(self):
        # Initialize the state
        self.x_initial = [0.0, 0.0, 0.0, 0.0]

    def winch_model(self, t, y, up):
        up_clipped = np.clip(up, -self.max_control_input, self.max_control_input)
        dy_dt = [
            self.A_11 * y[0] + self.B_11 * up_clipped,
            self.A_21 * y[0] + self.A_23 * y[2],
            self.A_32 * y[1] + self.A_33 * y[2],
            self.A_43 * y[2]
        ]
        return dy_dt

    def get_winch_rotational_velocity(self, dt, num_steps, up):
        t0 = 0  # Assuming the initial time is 0
        tf = dt * num_steps  # Final time = time step length * number of time steps
        time_series = np.linspace(t0, tf, num_steps)  # Generate a time series from t0 to tf with num_steps points

        tspan = (t0, tf)
        x_solution = solve_ivp(
            lambda t, y: self.winch_model(t, y, up),
            tspan,
            self.x_initial,
            t_eval=time_series
        )
        # print(f"calculated ivp at {num_steps} timestep")
        if len(x_solution.y) == 0:
            return 0.0
        else:
            return x_solution.y[2][-1]
