import numpy as np


class WaveGenerator:
    def __init__(
        self,
        coords,  # wrt to center of mass !!!
        frequency_1=0.0062,  # 0.1Hz
        frequency_2=0.0062,  # 0.1Hz
        num_points=101,
        timestep_delay=0.1,
        amplitude_1=1 / 4,
        amplitude_2=1 / 4,
    ):
        # Parameters
        self.frequency_1 = frequency_1  # frequency of 1st sine wave [rad/s]
        self.frequency_2 = frequency_2  # frequency of 2nd sine wave [rad/s]
        self.num_points = num_points  # number of points in each direction
        self.timestep_delay = timestep_delay  # delay in seconds for the second wave
        self.coords = coords
        self.amplitude_1 = amplitude_1
        self.amplitude_2 = amplitude_2
        # Grid of points
        self.x = np.linspace(-10, 10, self.num_points)
        self.y = np.linspace(-10, 10, self.num_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self._wave_plane = self.wave1(self.X, 0) + self.wave2(
            self.Y, 0, self.timestep_delay
        )
        self.wave = None

    def _bilin_intp(self, index: tuple, arr=None):
        """Calculate a value based on non-int array index using bilinear interpolation."""
        if arr is None:
            arr = self._wave_plane

        x_floor = int(np.floor(index[0]))
        x_ceil = int(np.ceil(index[0]))
        y_floor = int(np.floor(index[1]))
        y_ceil = int(np.ceil(index[1]))

        x_floor = max(x_floor, 0)
        x_ceil = min(x_ceil, arr.shape[0] - 1)
        y_floor = max(y_floor, 0)
        y_ceil = min(y_ceil, arr.shape[1] - 1)

        if x_floor == x_ceil and y_floor == y_ceil:
            return arr[x_floor, y_floor]

        top_left = arr[x_floor, y_floor]
        top_right = arr[x_floor, y_ceil]
        bottom_left = arr[x_ceil, y_floor]
        bottom_right = arr[x_ceil, y_ceil]

        x_weight = index[0] - x_floor
        y_weight = index[1] - y_floor

        # Perform bilinear interpolation
        top_interpolated = (top_right * x_weight) + (top_left * (1 - x_weight))
        bottom_interpolated = (bottom_right * x_weight) + (bottom_left * (1 - x_weight))
        interpolated_value = (bottom_interpolated * y_weight) + (
            top_interpolated * (1 - y_weight)
        )

        return interpolated_value

    def wave1(self, X, time):
        return self.amplitude_1 * np.sin(self.frequency_1 * (X + time * 2 * np.pi))

    def wave2(self, Y, time, delay):
        return self.amplitude_2 * np.sin(
            self.frequency_2 * (Y + (time - delay) * 2 * np.pi)
        )

    def coordinates_to_indices(self, coordinates: tuple) -> tuple:
        index_x = int(
            (coordinates[0] + self.x.min()) / np.round(self.x[1] - self.x[0], 1)
        )
        index_y = int(
            (coordinates[1] + self.y.min()) / np.round(self.y[1] - self.y[0], 1)
        )
        return index_x, index_y

    def update(self, dt):
        self._wave_plane = self.wave1(self.X, dt) + self.wave2(
            self.Y, dt, self.timestep_delay
        )
        idx_wh11 = self.coordinates_to_indices(
            (self.coords[0][0][0], self.coords[0][0][1])
        )
        idx_wh12 = self.coordinates_to_indices(
            (self.coords[0][1][0], self.coords[0][1][1])
        )
        idx_wh21 = self.coordinates_to_indices(
            (self.coords[1][0][0], self.coords[1][0][1])
        )
        idx_wh22 = self.coordinates_to_indices(
            (self.coords[1][1][0], self.coords[1][1][1])
        )

        self.wave = np.array(
            [
                [self._wave_plane[idx_wh11], self._wave_plane[idx_wh12]],
                [self._wave_plane[idx_wh21], self._wave_plane[idx_wh22]],
            ]
        )
        #              |self.wave[-dx,dy]    self.wave[dx,dy] |
        #  self.wave = |                                      |
        #              |self.wave[-dx,-dy]   self.wave[dx,-dy]|
        return self.wave
