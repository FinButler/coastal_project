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

# Bathymetry extraction

data = np.loadtxt("project_data/transect_cubic_spline_14750_entries.csv", delimiter=",", skiprows=1)

# Separate the distance and depth data
distances = data[:, 0]  # First column is distance (m)
depths = data[:, 1]     # Second column is depth (m)

# Convert to lists if needed
distance_list = distances.tolist()
depth_list = depths.tolist()
plotting_depth = []


for depth in depth_list:
    plot_depth = -1 * depth
    plotting_depth.append(plot_depth)


# Offshore Celerity Calculations
offshore_depth = 30

omega_s = 2*np.pi/avg_apd
omega_100 = 2*np.pi/max_periods

deep_group_cs = (9.81/(2*np.pi))*avg_apd
deep_group_c100 = (9.81/(2*np.pi))*max_periods

deep_cs = (9.81/(2*np.pi))*avg_apd
deep_c100 = (9.81/(2*np.pi))*max_periods

lambda_s = deep_cs/(avg_apd**-1)
lambda_100 = deep_cs/(max_periods**-1)

# Onshore Celerity Calculations

onshore_depth = depth_list
onshore_group_cs = []
onshore_group_c100 = []
onshore_cs = []
onshore_c100 = []

for depth in onshore_depth:
    shallow_group_cs = 0.5 * (
        1 + (4 * np.pi * (depth / lambda_s)) / np.sinh(4 * np.pi * (depth / lambda_s))
    ) * ((9.81 * avg_apd) / (2 * np.pi)) * np.tanh((2 * np.pi * depth) / lambda_s)
    onshore_group_cs.append(shallow_group_cs)

    shallow_group_c100 = 0.5 * (
        1 + (4 * np.pi * (depth / lambda_100)) / np.sinh(4 * np.pi * (depth / lambda_100))
    ) * ((9.81 * max_periods) / (2 * np.pi)) * np.tanh((2 * np.pi * depth) / lambda_100)
    onshore_group_c100.append(shallow_group_c100)

    on_cs = (lambda_s*(np.tanh(((omega_s**2)*depth/9.81)**0.75))**(2/3))/avg_apd
    onshore_cs.append(on_cs)

    on_c100 = (lambda_100 * (np.tanh(((omega_100 ** 2) * depth / 9.81) ** 0.75)) ** (2 / 3)) / max_periods
    onshore_c100.append(on_cs)

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

Kr_s = []
Kr_100 = []
theta_s = []
theta_100 = []

for oc in onshore_cs:
    siny_s = np.sin(y_deep)*oc/deep_cs
    Kref_s = ((1 - (np.sin(y_deep) ** 2)) / (1 - siny_s ** 2)) ** 0.25
    Kr_s.append(Kref_s)
    theta = np.degrees(np.arcsin(siny_s))
    theta_s.append(theta)

for oc in onshore_c100:
    siny_100 = np.sin(y_deep)*oc/deep_c100
    Kref_100 = ((1-(np.sin(y_deep)**2))/(1-siny_100**2))**0.25
    Kr_100.append(Kref_100)
    theta = np.degrees(np.arcsin(siny_100))
    theta_100.append(theta)

# Propagation Coefficients

sig_height = []
ret_height = []

for Ks, Kr in zip (Ks_s, Kr_s):
    onshore_height_s = significant_wvht*Ks*Kr
    sig_height.append(onshore_height_s)

for Ks, Kr in zip (Ks_100, Kr_100):
    onshore_height_100 = return_period*Ks*Kr
    ret_height.append(onshore_height_100)

# Breaking Limits

h_bs = (((deep_cs * significant_wvht**2) / (2 * (0.78**2) * (np.sqrt(9.81)))) * np.cos(np.degrees(15)))**(2/5)

h_b100 = (((deep_cs * return_period**2) / (2 * (0.78**2) * (np.sqrt(9.81)))) * np.cos(np.degrees(15)))**(2/5)

H_bs = []
H_b100 = []

for depth in onshore_depth:
    Hbs = 0.17*lambda_s*(1-np.exp(((-1.5*np.pi*depth)/(lambda_s))*(1+15*(np.tan(25/15000))**(4/3))))
    H_bs.append(Hbs)

    Hb100 = 0.17 * lambda_100 * (1 - np.exp(((-1.5 * np.pi * depth) / (lambda_100)) * (1 + 15 * (np.tan(25/15000)) ** (4 / 3))))
    H_b100.append(Hb100)

# Theta Differences

diff_s = [abs(theta_s[i+1] - theta_s[i]) for i in range(len(theta_s) - 1)]
diff_100 = [abs(theta_100[i+1] - theta_100[i]) for i in range(len(theta_100) - 1)]

approach_theta_s = np.concatenate(([65], np.cumsum(diff_s) + 65))
approach_theta_100 = np.concatenate(([65], np.cumsum(diff_100) + 65))

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot sig_height and ret_height on primary y-axis
ax1.plot(distance_list, sig_height, label="Significant Wave Height ($H_s$)", color="blue")
ax1.plot(distance_list, ret_height, label="Return Period Height ($H_{100}$)", color="green")
ax1.plot(distance_list, plotting_depth, label="Sea Depth", color="red")

# ax1.plot(onshore_depth, H_bs, label="Significant Wave Breaking Height (H_s)", color="blue", marker="x")
# ax1.plot(onshore_depth, H_b100, label="Return Period Breaking Height (H_100)", color="green", marker="x")
ax1.set_xlabel("Distance From Shoreline (m)")
ax1.set_ylabel("Height (m)")
ax1.set_xlim(max(distance_list), min(distance_list))
ax1.set_ylim(-30, 35)
ax1.legend(loc="upper left")

ax1.plot([230, 230], [-1.2, 5.1], color='blue', linestyle=':')  # Vertical line
ax1.plot([max(distance_list), 230], [5.1, 5.1], color='blue', linestyle=':')  # Horizontal line
ax1.scatter(230, 5.1, color='black', marker= "x", label=f'Critical Breaking Point')  # Marker at the top of the line

ax1.plot([550, 550], [-5.2, 16.1], color='green', linestyle=':')  # Vertical line
ax1.plot([max(distance_list), 550], [16.1, 16.1], color='green', linestyle=':')  # Horizontal line
ax1.scatter(550, 16.1, color='black', marker= "x",label=f'Point at (x=, y=)')  # Marker at the top of the line

ax1.plot([max(distance_list), min(distance_list)], [0, 0], color='lightblue', linestyle='--')

# Title and grid
plt.title("Evolution of Wave Heights as Waves Approach Coastline")
plt.grid(alpha=0.3)

# Show plot
plt.tight_layout()
plt.show()