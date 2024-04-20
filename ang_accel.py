import numpy as np


# Structure class to hold the properties
class Body(object):
    def __init__(self, mass, width, length, height):
        self.mass_boat = mass
        self.width = width
        self.length = length
        self.height = height
        self.Ixx = (self.mass_boat / 12) * (self.height**2 + self.width**2)
        self.Iyy = (self.mass_boat / 12) * (self.length**2 + self.width**2)
        self._force_applied_coords = np.array(
            [
                [
                    [-self.width / 2, self.length / 2, self.height / 2],  # top left
                    [self.width / 2, self.length / 2, self.height / 2],  # top right
                ],
                [
                    [-self.width / 2, -self.length / 2, self.height / 2],  # bottom left
                    [self.width / 2, -self.length / 2, self.height / 2],  # bottom right
                ],
            ],
            dtype=np.float64,
        )
        # Placeholder for angular_acceleration attribute
        self.angular_acceleration = None

    # Function to compute angular acceleration
    def calculate_angular_acceleration(self, forces):
        # Calculate torques due to forces at each corner
        torque_pitch = 0
        torque_roll = 0
        for i in range(2):
            for j in range(2):
                # Calculate the lever arm for the specified coordinate
                r = self._force_applied_coords[i, j]
                # The torque is r x F; here the force is perpendicular, so we take the magnitude directly
                torque_pitch += r[0] * forces[i, j]  # r[0] corresponds to the width, i.e., the lever arm for pitch
                torque_roll += r[1] * forces[i, j]  # r[1] corresponds to the length, i.e., the lever arm for roll

        # Sum torques from the top-view perspective, positive toward the body's forward and right
        total_torque = np.array([torque_pitch, torque_roll])

        # Calculate angular acceleration as torque divided by moment of inertia
        # Make sure to convert self.Ixx and self.Iyy into an array in the correct order
        self.angular_acceleration = total_torque / np.array([self.Ixx, self.Iyy])
        return self.angular_acceleration


# Instantiate Body object with example values
boat = Body(mass=1000, width=10, length=20, height=5)

# Assume some example forces
example_forces = np.array(
    [
        [500, 500],  # Force applied at the top left and top right corners
        [300, 300],  # Force applied at the bottom left and bottom right corners
    ]
)

# Calculate angular acceleration
angular_acceleration = boat.calculate_angular_acceleration(example_forces)
print("Angular acceleration (pitch, roll):", angular_acceleration)
