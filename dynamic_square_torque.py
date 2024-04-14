import numpy as np

class DynamicSquare:
    def __init__(self, mass, dimensions, timestep):
        self.mass = mass
        self.width = dimensions[0]
        self.length = dimensions[1]
        self.height = dimensions[2]
        self.I = mass * self.width * self.length / 6  # inertia for the square
        self.timestep = timestep

        # state variables
        self.position = np.array([0, 0, 0], dtype=np.float64)  # [x, y, z] [m]
        self.velocity = np.array([0, 0, 0], dtype=np.float64)  # [vx, vy, vz] [m/s]
        self.orientation = np.array([0, 0, 0], dtype=np.float64)  # [roll, pitch, yaw] [rad]
        self.angular_velocity = np.array([0, 0, 0], dtype=np.float64)  # [roll_rate, pitch_rate, yaw_rate] [rad/s]

        # force variables at the corners
        self.forces = np.zeros((4, 3))  # 4 corners with [fx, fy, fz] [N]
        self._torques = np.zeros((4, 3))
        self.total_torque = np.zeros(3)
        # top view, forward is up
        self._force_applied_coords = np.array([
            [self.width/4, self.length/4, self.height/2],  # bottom left
            [3*self.width/4, self.length/4, self.height/2],  # bottom right
            [3*self.width/4, 3*self.length/4, self.height/2],  # top right
            [self.width/4, 3*self.length/4, self.height/2],  # top left
        ], dtype=np.float64)

    def apply_external_force(self, corner_index, force):
        self.forces[corner_index] = force

    def calculate_torques(self):
        self._torques = np.cross(self._force_applied_coords - self.position, self.forces)
        self.total_torque = np.sum(self._torques, axis=0)
        return self.total_torque
    
    def simulate_step(self):
        # Calculate net forces
        net_force = np.sum(self.forces, axis=0)
        
        # Calculate net torques
        net_torque = self.calculate_torques()
        
        # Linear motion equations
        acceleration = net_force / self.mass
        self.velocity += acceleration * self.timestep
        self.position += self.velocity * self.timestep
        
        # Angular motion equations for roll and pitch
        angular_acceleration = net_torque / self.I
        self.angular_velocity += angular_acceleration * self.timestep
        self.orientation += self.angular_velocity * self.timestep
        
        # Reset the forces after each simulation step
        self.forces = np.zeros((4, 3))
        
    def simulate(self, num_steps):
        for _ in range(num_steps):
            self.simulate_step()


if __name__ == "__main__":
    # Example usage
    square = DynamicSquare(mass=500.0, dimensions=(4.0, 10.0, 2.0), timestep=0.01)
    # Apply a force of 10N upwards at the top right corner for 100 steps
    for _ in range(100):
        square.apply_external_force(corner_index=2, force=np.array([0, 0, 10]))
        square.simulate_step()

    # After applying forces, you can access the position and orientation
    print("Final Position:", square.position)
    print("Final Orientation:", square.orientation)