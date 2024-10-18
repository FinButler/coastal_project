import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data path
data_path = 'project_data/'

# List of years
dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# List to store all wave direction data from all years
all_wave_directions = []

# Loop over each year
for date in dates:
    # Load the data for the current year
    df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['MWD'])

    # Filter out NaN values and keep valid wave direction data
    df = df.dropna(subset=['MWD'])
    df_filtered = df[df['MWD'] != 999]  # Remove any invalid placeholder data like 999

    # Append the wave directions for the current year to the overall list
    all_wave_directions.extend(df_filtered['MWD'].tolist())

# Convert the list of all wave directions to a numpy array
all_wave_directions = np.array(all_wave_directions)

# Convert wave directions from degrees to radians for polar plotting
all_wave_directions_rad = np.deg2rad(all_wave_directions)

# Create bins for wave direction (e.g., 36 bins of 10 degrees each)
n_bins = 36
hist, bin_edges = np.histogram(all_wave_directions_rad, bins=n_bins, range=(0, 2*np.pi))

# Prepare the rose diagram
angles = np.linspace(0, 2*np.pi, n_bins, endpoint=False)
widths = (2 * np.pi) / n_bins  # Width of each bin

# Plot the rose diagram
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# Use bar plot to create the rose diagram
bars = ax.bar(angles, hist, width=widths, bottom=0.0, color='blue', edgecolor='black', alpha=0.75)

# Set the direction of the 0 degree line to match compass directions (north-up)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)  # Clockwise

# Add labels
ax.set_title('Wave Direction Rose Diagram (MWD)', fontsize=16)

# Display the plot
plt.show()
