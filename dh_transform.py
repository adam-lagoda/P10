import numpy as np # Scientific computing library

def calculate_dh_rotation_matrice(
    theta_1: np.float64,  # Joint 1 [rad]
    d1: np.float64,  # Displacement of link 2 [m]
    d2: np.float64,  # Displacement of link 9 [m]
    initial_boat_rope_length: np.float64,  # [m]
    intital_load_rope_length: np.float64,  # [m]
    ):
    # theta_2 = 0  # Joint 2
    theta_2 = np.pi/2 - theta_1  # Joint 2
    theta_4 = 0  # Joint 4 0 
    theta_5 = np.pi/2  # Joint 5 90
    theta_6 = 0  # Joint 6 0
    theta_3 = 0  # Joint 3 0
    theta_7 = 0  # Joint 7 0
    theta_9 = 0  # Joint 9 0
    theta_8 = 0  # Joint 8 0

    # Link lengths in meters
    a1 = initial_boat_rope_length  #  Length of link 1
    a2 = 0  # Length of link 2
    a3 = 0  # Length of link 3 (should be zero)
    a4 = 0  # Length of link 4 (should be zero)
    a5 = 0.1  # Length of link 5
    a6 = 0  # Length of link 6 (should be zero)
    a7 = 0  # Length of link 7 (should be zero)
    a8 = intital_load_rope_length  # Length of link 8
    a9 = 0  # Length of link 9

    
    # Declare the Denavit-Hartenberg table. 
    # It will have four columns, to represent:
    # theta, alpha, a, and d
    # We have the convert angles to radians.
    d_h_table = np.array(
        [
            [theta_1                                , np.deg2rad(90)    , 0 , 0         ],
            [theta_2                                , np.deg2rad(-90)   , 0 , a1+a2+d1  ],
            [np.deg2rad(theta_3 + np.deg2rad(90))   , np.deg2rad(90)    , 0 , 0         ],
            [np.deg2rad(theta_4 + np.deg2rad(-90))  , np.deg2rad(90)    , 0 , 0         ],
            [theta_5                                , np.deg2rad(90)    , a5, 0         ],
            [np.deg2rad(theta_6 + np.deg2rad(90))   , np.deg2rad(90)    , 0 , 0         ],
            [np.deg2rad(theta_7 + np.deg2rad(90))   , np.deg2rad(-90)   , 0 , 0         ],
            [np.deg2rad(theta_8)                    , 0                 , 0 , a8+a9+d2  ],
            [np.deg2rad(theta_9)                    , 0                 , 0 , 0         ]
        ]
    ) 
    
    # Homogeneous transformation matrix from frame 0 to frame 1
    i = 0
    homgen_0_1 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])  
    
    # Homogeneous transformation matrix from frame 1 to frame 2
    i = 1
    homgen_1_2 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])  
    
    # Homogeneous transformation matrix from frame 2 to frame 3
    i = 2
    homgen_2_3 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])  
    
    # Homogeneous transformation matrix from frame 3 to frame 4
    i = 3
    homgen_3_4 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])  
    
    # Homogeneous transformation matrix from frame 4 to frame 5
    i = 4
    homgen_4_5 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])  
    # Homogeneous transformation matrix from frame 5 to frame 6
    i = 5
    homgen_5_6 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])
    # Homogeneous transformation matrix from frame 6 to frame 7
    i = 6
    homgen_6_7 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])
    # Homogeneous transformation matrix from frame 7 to frame 8
    i = 7
    homgen_7_8 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])
    # Homogeneous transformation matrix from frame 8 to frame 9
    i = 8
    homgen_8_9 = np.array([[np.cos(d_h_table[i,0]), -np.sin(d_h_table[i,0]) * np.cos(d_h_table[i,1]), np.sin(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.cos(d_h_table[i,0])],
                        [np.sin(d_h_table[i,0]), np.cos(d_h_table[i,0]) * np.cos(d_h_table[i,1]), -np.cos(d_h_table[i,0]) * np.sin(d_h_table[i,1]), d_h_table[i,2] * np.sin(d_h_table[i,0])],
                        [0, np.sin(d_h_table[i,1]), np.cos(d_h_table[i,1]), d_h_table[i,3]],
                        [0, 0, 0, 1]])
    
    # homogeneous transformation matrices 0 -> 9
    homgen_0_9 = homgen_0_1 @ homgen_1_2 @ homgen_2_3 @ homgen_3_4 @ homgen_4_5 @ homgen_5_6 @ homgen_6_7 @ homgen_7_8 @ homgen_8_9 

    #┌ Rx  Rz  Ry     ┐
    #│ R00 R01 R02 Tx │ 
    #│ R10 R11 R12 Ty │ 
    #│ R20 R21 R22 Tz │ 
    #│ 0   0   0   1  │
    #└                ┘
    return homgen_0_9