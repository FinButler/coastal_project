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

use_situ = True

if use_situ == True:

    for date in dates:

        df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['WVHT', 'APD'])
        df = df.dropna(subset=['WVHT', 'APD'])
        df_filtered = df[(df['WVHT'] != 99) & (df['APD'] != 99)]  # Apply both filters

        # Append the wave heights and average periods for the current year to the respective lists
        all_wave_heights.extend(df_filtered['WVHT'].tolist())
        all_periods.extend(df_filtered['APD'].tolist())

elif use_situ == False:

    df = pd.read_csv("project_data/off_sig_h.csv", skiprows=7)
    # Check and strip any whitespace from column names
    df.columns = df.columns.str.strip()

    all_wave_heights.extend(df['VHM0_WW'].dropna().tolist())
    all_periods.extend(df['VTM01_WW'].dropna().tolist())

    all_wave_heights = pd.Series(all_wave_heights)
    all_periods = pd.Series(all_periods)

all_wave_heights = pd.Series(all_wave_heights)
all_periods = pd.Series(all_periods)

# Sort the wave height data to calculate the top one-third of wave heights
all_wave_heights_sorted = all_wave_heights.sort_values(ascending=False)
top_one_third_count = int(len(all_wave_heights_sorted) / 3)
top_one_third_waves = all_wave_heights_sorted[:top_one_third_count]

# Find periods relating to significant waves
top_one_third_periods = all_periods[top_one_third_waves.index]

# Calculate Period of all extreme waves
wave_heights_above_4m = all_wave_heights[all_wave_heights > 4]
periods_above_4m = all_periods[wave_heights_above_4m.index]
max_periods = periods_above_4m.mean()



# Calculate the overall statistics for wave heights
avg_wvht = all_wave_heights.mean()
rms_wvht = np.sqrt(np.mean(all_wave_heights**2))
significant_wvht = top_one_third_waves.mean()
max_wvht = all_wave_heights.max()
avg_apd = top_one_third_periods.mean()
# return_period = np.sqrt(np.log(avg_apd/(100*365*24*3600))*(-(significant_wvht**2))/2)
return_period = 10.84


print(f'Average Significant Wave Period: {avg_apd:.2f} seconds')
print(f'100 Year Return Period: {return_period:.2f} m')
print(f"Significant Wave Height (H_s): {significant_wvht:.2f}")
print(f'Average Wave Period for Heights > 4m: {max_periods:.2f} seconds')

# Offshore Celerity Calculations
offshore_depth = 19

lambda_s = (9.81*avg_apd**2)/(2*np.pi)
lambda_100 = (9.81*max_periods**2)/(2*np.pi)

omega_s = 2*np.pi/avg_apd
omega_100 = 2*np.pi/max_periods

deep_group_cs = (9.81/(2*np.pi))*avg_apd
deep_group_c100 = (9.81/(2*np.pi))*max_periods

deep_cs = (9.81/(2*np.pi))*avg_apd
deep_c100 = (9.81/(2*np.pi))*max_periods

# Onshore Celerity Calculations

onshore_depth = list(range(19, 0, -1))
onshore_group_cs = []
onshore_group_c100 = []


for depth in onshore_depth:
    shallow_group_cs = 0.5 * (
        1 + (4 * np.pi * (depth / lambda_s)) / np.sinh(4 * np.pi * (depth / lambda_s))
    ) * ((9.81 * avg_apd) / (2 * np.pi)) * np.tanh((2 * np.pi * depth) / lambda_s)
    onshore_group_cs.append(shallow_group_cs)

    shallow_group_c100 = 0.5 * (
        1 + (4 * np.pi * (depth / lambda_100)) / np.sinh(4 * np.pi * (depth / lambda_100))
    ) * ((9.81 * max_periods) / (2 * np.pi)) * np.tanh((2 * np.pi * depth) / lambda_100)
    onshore_group_c100.append(shallow_group_c100)

onshore_cs = lambda_s/avg_apd
onshore_c100 = lambda_100/max_periods

# Shoaling Coefficients
Ks_s = []
Ks_100 = []

for cs in onshore_group_cs:
    Ks_sig = np.sqrt(deep_group_cs/cs)
    Ks_s.append(Ks_sig)

for c100 in onshore_group_c100:
    Ks_ret = np.sqrt(deep_group_c100/c100)
    Ks_100.append(Ks_ret)


# Refraction Coefficients

shore_angle = 80
wave_angle = 65

y_deep = shore_angle - wave_angle



print("Significant Omega: " + str(omega_s))
print("100 Omega: " + str(omega_100))
print("Significant lambda: " + str(lambda_s))
print("100 lambda: " + str(lambda_100))
print("Significant cg offshore: " + str(deep_group_cs))
print("100 cg offshore: " + str(deep_group_c100))

print(Ks_s)
print(Ks_100)

print(y_deep)