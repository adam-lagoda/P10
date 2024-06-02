# Active heave compensation using Deep Reinforcement Learning
## Master Thesis at Aalborg University
## Esbjerg, Denmark 2023-2024

### Structure:
```bash
.
├── README.md
├── buoyantboat
│   ├── env
│   │   ├── __init__.py
│   │   ├── boat_env.py  # Main file contatining model equations and calulcations
│   │   ├── dh_transform.py  # DH Transformation Equations
│   │   ├── wave_generator.py  # Synthetic Wave Generator
│   │   └── winch.py  # Model of the Hydraulic Winch
│   ├── simulate
│   │   └── simulate.py  # Script for simulating the system with no control input
│   ├── train
│   │   ├── boat_ddpg.py  # Training script for DDPG
│   │   ├── boat_dqn.py  # Training script for DQN
│   │   ├── boat_ppo.py  # Training script for PPO
│   │   ├── boat_sac.py  # Training script for SAC
│   │   └── boat_td3.py  # Training script for TD3
│   └── validate
│       ├── ddpg_test.py  # Validation script for DDPG
│       ├── dqn_test.py  # Validation script for DQN
│       ├── ppo_test.py  # Validation script for PPO
│       └── sac_test.py  # Validation script for SAC
├── environment.yml  # Micromamba environment file
├── misc
│   ├── r2  # R^2 Analysis Scripts
│   └── wavegen.py  # Animated Wave Generation script
├── trained_policies
│   └── converged  # Trained and converged policies with network files and logs
└── wave_prediction
    ├── wave_prediction_cnn.py  # Wave prediction using CNN
    └── wave_prediction_lstm.py  # Wave prediction using LSTM
```

### Authors:
1.  Adam Lagoda
2.  Thomas Aagaard Pedersen
3.  Seyedeh Fatemeh Mahdavi Sharifi


