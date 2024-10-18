import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data path
data_path = 'project_data/'

# List of years
dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# List to store all wave height data from all years
all_wave_heights = []

# Loop over each year
for date in dates:
    # Load the data for the current year
    df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['WVHT'])

    # Filter out NaN values and remove the number 99
    df = df.dropna(subset=['WVHT'])
    df_filtered = df[df['WVHT'] != 99]

    # Append the wave heights for the current year to the overall list
    all_wave_heights.extend(df_filtered['WVHT'].tolist())

# Convert the list of all wave heights to a pandas Series
all_wave_heights = pd.Series(all_wave_heights)

# Sort the data to calculate the top one-third of wave heights
all_wave_heights_sorted = all_wave_heights.sort_values(ascending=False)
top_one_third_count = int(len(all_wave_heights_sorted) / 3)
top_one_third_waves = all_wave_heights_sorted[:top_one_third_count]

# Calculate the overall statistics
overall_avg_wvht = all_wave_heights.mean()
overall_rms_wvht = np.sqrt(np.mean(all_wave_heights**2))
overall_significant_wvht = top_one_third_waves.mean()
overall_max_wvht = all_wave_heights.max()

# Prepare data for the bar chart
categories = ['Average Wave Height (H_z)', 'RMS Wave Height (H_rms)',
              'Significant Wave Height (H_s)', 'Maximum Wave Height (H_max)']
values = [overall_avg_wvht, overall_rms_wvht, overall_significant_wvht, overall_max_wvht]

# Plot the horizontal bar chart
plt.figure(figsize=(8, 6))
bars = plt.barh(categories, values, color=['blue', 'green', 'red', 'purple'])

# Add values to each bar
for bar in bars:
    plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.2f}', va='center')

# Labels and title
plt.xlabel('Wave Height (meters)')
plt.title('Wave Height Statistics Averaged Over All Years')

# Display the plot
plt.show()
