import pandas as pd
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('./buoyantboat/validate/ObjectState.csv')

# Extract the 'z' column into an array
z_array = df['z'].values
time = df['t'].values

# Normalize the 'z' array between -1 and 1
z_min = np.min(z_array)
z_max = np.max(z_array)
z_normalized = 2 * (z_array - z_min) / (z_max - z_min) - 1

# Create a new dataframe from time and z_normalized
new_df = pd.DataFrame({'time': time[:2000], 'z_normalized': z_normalized[:2000]})

# Save the new dataframe to a csv file
new_df.to_csv('./buoyantboat/validate/wave_data_ue.csv', index=False)