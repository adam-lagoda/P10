import numpy as np
import pandas as pd

# Load the UE Object States from the CSV file
df = pd.read_csv("./buoyantboat/validate/ObjectState.csv")

z_array = df["z"].values
time = df["t"].values

z_min = np.min(z_array)
z_max = np.max(z_array)
z_normalized = 2 * (z_array - z_min) / (z_max - z_min) - 1

new_df = pd.DataFrame({"time": time[:2000], "z_normalized": z_normalized[:2000]})

new_df.to_csv("./buoyantboat/validate/wave_data_ue.csv", index=False)
