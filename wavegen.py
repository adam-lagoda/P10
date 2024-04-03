import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
frequency_1 = 1  # frequency of 1st sine wave
frequency_2 = 1  # frequency of 2nd sine wave
num_points = 100  # number of points in each direction
time_duration = 1  # duration of the animation in seconds
frames_per_second = 24  # frames per second in the animation
timestep_delay = 0.1  # delay in seconds for the second wave

# Create a grid of points
x = np.linspace(-np.pi, np.pi, num_points)
y = np.linspace(-np.pi, np.pi, num_points)
X, Y = np.meshgrid(x, y)

# Wave functions
def wave1(X, time):
    return 1/2 * np.sin(frequency_1 * (X + time * 2 * np.pi))

def wave2(Y, time, delay):
    return 1/2 * np.sin(frequency_2 * (Y + (time - delay) * 2 * np.pi))

# Initialize figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize the surface plot
wave = wave1(X, 0) + wave2(Y, 0, timestep_delay)
surf = ax.plot_surface(X, Y, wave, cmap='viridis')

# Update function for animation
def update(frame):
    time = frame / frames_per_second
    wave = wave1(X, time) + wave2(Y, time, timestep_delay)
    ax.clear()
    surf = ax.plot_surface(X, Y, wave, cmap='viridis')
    return surf,

# Create animation
ani = FuncAnimation(fig, update, frames=time_duration * frames_per_second, blit=False, interval=1000 / frames_per_second, repeat=True)

# Display animation
plt.show()