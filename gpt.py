import numpy as np
import matplotlib.pyplot as plt

# Constants
mass = 1.0  # Mass of the box (kg)
length = 1.0  # Length of the box (m)
height = 1.0  # Height of the box (m)
width = 1.0  # Width of the box (m)
Ixx = (mass / 12) * (height**2 + width**2)  # Moment of inertia about x-axis (kg m^2)
Iyy = (mass / 12) * (length**2 + width**2)  # Moment of inertia about y-axis (kg m^2)
Izz = (mass / 12) * (length**2 + height**2)  # Moment of inertia about z-axis (kg m^2)

# Initial conditions
omega = np.array([0.0, 0.0, 0.0])  # Initial angular velocity (rad/s)
theta = np.array([0.0, 0.0, 0.0])  # Initial orientation (roll, pitch, yaw) (rad)

# Time parameters
dt = 0.01  # Time step (s)
total_time = 10.0  # Total simulation time (s)
num_steps = int(total_time / dt)  # Number of time steps

# Forces applied at the corners (N)
forces = np.array([[1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1]])

# Arrays to store results
angular_velocities = np.zeros((num_steps, 3))
orientations = np.zeros((num_steps, 3))

# Simulation loop
for i in range(num_steps):
    # Calculate torque
    torque = np.cross(forces, length / 2 * np.array([[1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]]))

    # Calculate angular acceleration using Euler's equations
    alpha = np.array([torque[0, 0] / Ixx, torque[1, 1] / Iyy, torque[2, 2] / Izz])

    # Update angular velocity and orientation using Euler integration
    omega += alpha * dt
    theta += omega * dt

    # Store results
    angular_velocities[i] = omega
    orientations[i] = theta

# Extract roll and pitch angles
roll = orientations[:, 0]
pitch = orientations[:, 1]

# Plot results
time = np.arange(0, total_time, dt)
plt.plot(time, roll, label="Roll")
plt.plot(time, pitch, label="Pitch")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Roll and Pitch angles over time")
plt.legend()
plt.grid(True)
plt.show()
