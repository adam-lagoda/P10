import numpy as np


def calculate_dh_rotation_matrice(
    _theta: np.float64,  # Joint 1 [rad]
    boat_side_rope_length: np.float64,  # [m]
    load_side_rope_length: np.float64,  # [m]
):
    # Link lengths in meters
    a1 = boat_side_rope_length  # Length of link 1
    a3 = 1  # Length of link 3
    a5 = load_side_rope_length  # Length of link 5

    # Initialize values for the joint angles (radians)
    theta_1 = np.pi  # Joint 1
    theta_2 = -_theta  # Joint 2 wrt to vertical z axis
    theta_3 = 0  # Joint 3
    theta_4 = np.pi / 2 + theta_2  # Joint 4
    theta_5 = np.deg2rad(10)  # Joint 5
    theta_6 = 0  # Joint 6

    # Declare the Denavit-Hartenberg table.
    # It will have four columns, to represent:
    # theta, alpha, r, and d
    # We have the convert angles to radians.
    d_h_table = np.array(
        [
            [theta_1, np.deg2rad(90), 0, 0],
            [theta_2, np.deg2rad(-90), 0, 0],
            [theta_3, np.deg2rad(-90), 0, a1],
            [theta_4, 0, a3, 0],
            [theta_5, np.deg2rad(-90), 0, 0],
            [theta_6, 0, 0, a5],
        ]
    )

    # Homogeneous transformation matrix from frame 0 to frame 1
    i = 0
    homgen_0_1 = np.array(
        [
            [
                np.cos(d_h_table[i, 0]),
                -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.cos(d_h_table[i, 0]),
            ],
            [
                np.sin(d_h_table[i, 0]),
                np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.sin(d_h_table[i, 0]),
            ],
            [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
            [0, 0, 0, 1],
        ]
    )
    i = 1
    homgen_1_2 = np.array(
        [
            [
                np.cos(d_h_table[i, 0]),
                -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.cos(d_h_table[i, 0]),
            ],
            [
                np.sin(d_h_table[i, 0]),
                np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.sin(d_h_table[i, 0]),
            ],
            [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
            [0, 0, 0, 1],
        ]
    )

    # Homogeneous transformation matrix from frame 1 to frame 2
    i = 2
    homgen_2_3 = np.array(
        [
            [
                np.cos(d_h_table[i, 0]),
                -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.cos(d_h_table[i, 0]),
            ],
            [
                np.sin(d_h_table[i, 0]),
                np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.sin(d_h_table[i, 0]),
            ],
            [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
            [0, 0, 0, 1],
        ]
    )

    # Homogeneous transformation matrix from frame 2 to frame 3
    i = 3
    homgen_3_4 = np.array(
        [
            [
                np.cos(d_h_table[i, 0]),
                -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.cos(d_h_table[i, 0]),
            ],
            [
                np.sin(d_h_table[i, 0]),
                np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.sin(d_h_table[i, 0]),
            ],
            [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
            [0, 0, 0, 1],
        ]
    )

    # Homogeneous transformation matrix from frame 3 to frame 4
    i = 4
    homgen_4_5 = np.array(
        [
            [
                np.cos(d_h_table[i, 0]),
                -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.cos(d_h_table[i, 0]),
            ],
            [
                np.sin(d_h_table[i, 0]),
                np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.sin(d_h_table[i, 0]),
            ],
            [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
            [0, 0, 0, 1],
        ]
    )

    # Homogeneous transformation matrix from frame 4 to frame 5
    i = 5
    homgen_5_6 = np.array(
        [
            [
                np.cos(d_h_table[i, 0]),
                -np.sin(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                np.sin(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.cos(d_h_table[i, 0]),
            ],
            [
                np.sin(d_h_table[i, 0]),
                np.cos(d_h_table[i, 0]) * np.cos(d_h_table[i, 1]),
                -np.cos(d_h_table[i, 0]) * np.sin(d_h_table[i, 1]),
                d_h_table[i, 2] * np.sin(d_h_table[i, 0]),
            ],
            [0, np.sin(d_h_table[i, 1]), np.cos(d_h_table[i, 1]), d_h_table[i, 3]],
            [0, 0, 0, 1],
        ]
    )

    homgen_0_6 = homgen_0_1 @ (
        homgen_1_2 @ (homgen_2_3 @ (homgen_3_4 @ (homgen_4_5 @ homgen_5_6)))
    )
    homgen_0_6 = np.round(homgen_0_6, decimals=3)
    # ┌ Rx  Rz  Ry     ┐
    # │ R00 R01 R02 Tx │
    # │ R10 R11 R12 Ty │
    # │ R20 R21 R22 Tz │
    # │ 0   0   0   1  │
    # └                ┘
    return homgen_0_6
