import numpy as np
import pandas as pd


def r2_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    # Calculate the mean of the observed time series
    y_mean = np.mean(y_true)

    # Calculate the total sum of squares
    ss_tot = np.sum((y_true - y_mean) ** 2)

    # Calculate the residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate the R^2 value
    r2 = 1 - (ss_res / ss_tot)

    return r2


df_sac = pd.read_csv("./misc/r2/states_sac_no_winch.csv")
df_dqn = pd.read_csv("./misc/r2/states_dqn_no_winch.csv")
df_ddpg = pd.read_csv("./misc/r2/states_ddpg_no_winch.csv")
df_ppo = pd.read_csv("./misc/r2/states_ppo_no_winch.csv")

sac_l_v = df_sac["Load_Velocity_Z"].values
sac_l_v_t = df_sac["Target_Load_Velocity_Z"].values
sac_l_p = df_sac["Load_Position_Z"].values
sac_l_p_t = df_sac["Target_Load_Position_Z"].values
sac_max_v = max(df_sac["Load_Velocity_Z"].values[:250])
sac_min_v = min(df_sac["Load_Velocity_Z"].values[:250])
sac_max_p = max(df_sac["Load_Position_Z"].values[:250])
sac_min_p = min(df_sac["Load_Position_Z"].values[:250])
sac_v_err = abs(sac_max_v - sac_min_v)
sac_p_err = abs(sac_max_p - sac_min_p)


dqn_l_v = df_dqn["Load_Velocity_Z"].values
dqn_l_v_t = df_dqn["Target_Load_Velocity_Z"].values
dqn_l_p = df_dqn["Load_Position_Z"].values
dqn_l_p_t = df_dqn["Target_Load_Position_Z"].values
dqn_max_v = max(df_dqn["Load_Velocity_Z"].values[:250])
dqn_min_v = min(df_dqn["Load_Velocity_Z"].values[:250])
dqn_max_p = max(df_dqn["Load_Position_Z"].values[:250])
dqn_min_p = min(df_dqn["Load_Position_Z"].values[:250])
dqn_v_err = abs(dqn_max_v - dqn_min_v)
dqn_p_err = abs(dqn_max_p - dqn_min_p)


ddpg_l_v = df_ddpg["Load_Velocity_Z"].values
ddpg_l_v_t = df_ddpg["Target_Load_Velocity_Z"].values
ddpg_l_p = df_ddpg["Load_Position_Z"].values
ddpg_l_p_t = df_ddpg["Target_Load_Position_Z"].values
ddpg_max_v = max(df_ddpg["Load_Velocity_Z"].values[:250])
ddpg_min_v = min(df_ddpg["Load_Velocity_Z"].values[:250])
ddpg_max_p = max(df_ddpg["Load_Position_Z"].values[:250])
ddpg_min_p = min(df_ddpg["Load_Position_Z"].values[:250])
ddpg_v_err = abs(ddpg_max_v - ddpg_min_v)
ddpg_p_err = abs(ddpg_max_p - ddpg_min_p)


ppo_l_v = df_ppo["Load_Velocity_Z"].values
ppo_l_v_t = df_ppo["Target_Load_Velocity_Z"].values
ppo_l_p = df_ppo["Load_Position_Z"].values
ppo_l_p_t = df_ppo["Target_Load_Position_Z"].values
ppo_max_v = max(df_ppo["Load_Velocity_Z"].values[:250])
ppo_min_v = min(df_ppo["Load_Velocity_Z"].values[:250])
ppo_max_p = max(df_ppo["Load_Position_Z"].values[:250])
ppo_min_p = min(df_ppo["Load_Position_Z"].values[:250])
ppo_v_err = abs(ppo_max_v - ppo_min_v)
ppo_p_err = abs(ppo_max_p - ppo_min_p)

# Calculate the R^2 value between the two time series
r2_sac_vel = r2_score(sac_l_v, sac_l_v_t)
r2_sac_pos = r2_score(sac_l_p, sac_l_p_t)
r2_dqn_vel = r2_score(dqn_l_v, dqn_l_v_t)
r2_dqn_pos = r2_score(dqn_l_p, dqn_l_p_t)
r2_ddpg_vel = r2_score(ddpg_l_v, ddpg_l_v_t)
r2_ddpg_pos = r2_score(ddpg_l_p, ddpg_l_p_t)
r2_ppo_vel = r2_score(ppo_l_v, ppo_l_v_t)
r2_ppo_pos = r2_score(ppo_l_p, ppo_l_p_t)

# print(f"R^2 value SAC velocity: {r2_sac_vel}")
# print(f"R^2 value SAC position: {r2_sac_pos}")
# print(f"Error SAC velocity: {sac_v_err}")
# print(f"Error SAC position: {sac_p_err}")
# print(f"R^2 value DQN velocity: {r2_dqn_vel}")
# print(f"R^2 value DQN position: {r2_dqn_pos}")
# print(f"Error DQN velocity: {dqn_v_err}")
# print(f"Error DQN position: {dqn_p_err}")
# Create a DataFrame with the results
results = {
    "Metric": [
        "R^2 value velocity",
        "R^2 value position",
        "Error velocity",
        "Error position",
    ],
    "SAC": [r2_sac_vel, r2_sac_pos, sac_v_err, sac_p_err],
    "DQN": [r2_dqn_vel, r2_dqn_pos, dqn_v_err, dqn_p_err],
    "DDGN": [r2_ddpg_vel, r2_ddpg_pos, ddpg_v_err, ddpg_p_err],
    "PPO": [r2_ppo_vel, r2_ppo_pos, ppo_v_err, ppo_p_err],
}

results_df = pd.DataFrame(results)

# Print the results DataFrame
print(results_df)
