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


df_sac = pd.read_csv("./misc/r2/states_sac_no_winch_ue_data.csv")
df_dqn = pd.read_csv("./misc/r2/states_dqn_no_winch_ue_data.csv")

sac_l_v = df_sac["Load_Velocity_Z"].values
sac_l_v_t = df_sac["Target_Load_Velocity_Z"].values
sac_l_p = df_sac["Load_Position_Z"].values
sac_l_p_t = df_sac["Target_Load_Position_Z"].values
sac_max_v = max(df_sac["Load_Velocity_Z"].values[250:400])
sac_min_v = min(df_sac["Load_Velocity_Z"].values[250:400])
sac_max_p = max(df_sac["Load_Position_Z"].values[250:400])
sac_min_p = min(df_sac["Load_Position_Z"].values[250:400])
sac_v_err = abs(sac_max_v - sac_min_v)
sac_p_err = abs(sac_max_p - sac_min_p)


dqn_l_v = df_dqn["Load_Velocity_Z"].values
dqn_l_v_t = df_dqn["Target_Load_Velocity_Z"].values
dqn_l_p = df_dqn["Load_Position_Z"].values
dqn_l_p_t = df_dqn["Target_Load_Position_Z"].values
dqn_max_v = max(df_dqn["Load_Velocity_Z"].values[250:400])
dqn_min_v = min(df_dqn["Load_Velocity_Z"].values[250:400])
dqn_max_p = max(df_dqn["Load_Position_Z"].values[250:400])
dqn_min_p = min(df_dqn["Load_Position_Z"].values[250:400])
dqn_v_err = abs(dqn_max_v - dqn_min_v)
dqn_p_err = abs(dqn_max_p - dqn_min_p)


# Calculate the R^2 value between the two time series
r2_sac_vel = r2_score(sac_l_v, sac_l_v_t)
r2_sac_pos = r2_score(sac_l_p, sac_l_p_t)
r2_dqn_vel = r2_score(dqn_l_v, dqn_l_v_t)
r2_dqn_pos = r2_score(dqn_l_p, dqn_l_p_t)

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
}

results_df = pd.DataFrame(results)

# Print the results DataFrame
print(results_df)
