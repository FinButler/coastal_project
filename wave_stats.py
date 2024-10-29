import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data path
data_path = 'project_data/'

# List of years
dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Lists to store all wave height and significant period data from all years
all_wave_heights = []
all_periods = []

# Loop over each year
for date in dates:

    df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['WVHT', 'APD'])
    df = df.dropna(subset=['WVHT', 'APD'])
    df_filtered = df[(df['WVHT'] != 99) & (df['APD'] != 99)]  # Apply both filters

    # Append the wave heights and average periods for the current year to the respective lists
    all_wave_heights.extend(df_filtered['WVHT'].tolist())
    all_periods.extend(df_filtered['APD'].tolist())

all_wave_heights = pd.Series(all_wave_heights)
all_periods = pd.Series(all_periods)

# Sort the wave height data to calculate the top one-third of wave heights
all_wave_heights_sorted = all_wave_heights.sort_values(ascending=False)
top_one_third_count = int(len(all_wave_heights_sorted) / 3)
top_one_third_waves = all_wave_heights_sorted[:top_one_third_count]

# Find periods relating to significant waves
top_one_third_periods = all_periods[top_one_third_waves.index]

# Calculate the overall statistics for wave heights
avg_wvht = all_wave_heights.mean()
rms_wvht = np.sqrt(np.mean(all_wave_heights**2))
significant_wvht = top_one_third_waves.mean()
max_wvht = all_wave_heights.max()
avg_apd = top_one_third_periods.mean()
return_period = np.sqrt(np.log(avg_apd/(100*365*24*3600))*(-(significant_wvht**2))/2)


print(f'Average Significant Wave Period: {avg_apd:.2f} seconds')
print(f'100 Year Return Period: {return_period:.2f} m')
print(f"Significant Wave Height (H_s): {significant_wvht:.2f}")

# Check that periods correspond to significant waves
# print(top_one_third_waves.head())
# print(top_one_third_periods.head())

categories = ['H_z', 'H_rms',
              'H_s', 'H_max', 'H_100' ]
values = [avg_wvht, rms_wvht, significant_wvht, max_wvht, return_period]

plt.figure(figsize=(8, 6))
bars = plt.barh(categories, values, color=['blue', 'green', 'red', 'purple', 'orange'])


for bar in bars:
    plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.2f}', va='center')


plt.xlabel('Wave Height (m)')
plt.title('Wave Height Stats Over All Years')

plt.show()
