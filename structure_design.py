import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data extraction from previous scripts
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

# Wave propagation from previous scripts
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

# Structural Dimensions
h = 5
h_c = 2
B = 2

Hs = 3
H100 =16.1

S = 2

K_D = 3.5

# Rock armour - Trunk

rho_r = 2600

rho_w = 1029

delta = (rho_r/rho_w) - 1

alpha = 1/5

M_50 = (rho_r * Hs ** 3)/(K_D * (1/np.tan(alpha)) * delta ** 3) # Hudson Formula

Dn_50 = (-0.14 * Hs * (Hs / lambda_s) ** (-1/3)) / (delta * np.log((h_c/h) / (2.1 + 0.1 * S)))

N_s = ((K_D * (1 / np.tan(alpha))) ** (1/3))/1.27

N = Hs/(delta * Dn_50)

G = 1.4 # for rock

p = 0.5

N_od = G * (1 - p) * S

# print(N_s)
# print(N)

# Armour Stability using Eurocodes

P = 0.3
n = 4000

surf = (np.tan(alpha))/(np.sqrt(2 * np.pi * Hs / (9.81 * avg_apd ** 2)))

Euro_D50_breaking = (Hs * np.sqrt(surf)) / ((delta) * (6.2 * (P ** 0.18)) * (S / (np.sqrt(n))) ** 0.2) # breaking
Euro_D50_surging = (Hs) / ((delta) * (P ** -0.13) * ((S / (np.sqrt(n))) ** 0.2) * np.sqrt((1 / np.tan(alpha))) * (surf ** P))

Euro_N = Hs / (delta * Euro_D50_breaking)

print(surf)

print(Euro_D50_surging)
print(Euro_N)
print(N_s)


# Toe Stability - Trunk

Dn_50_t = 0.5 * Dn_50

h_t = h - Dn_50_t

Bt = 1

s = 2 * np.pi * Hs / (9.81 * avg_apd ** 2)

N_od = 0.5

Ns_t = 1.2 + 11.2 * ((h_t/h) ** (7/4)) * ((Bt/Hs) ** (-1/10)) * (s ** (1/6)) * (N_od ** (2/5)) # Etemad-Shahidi

# Overtopping Calcs

R_c = h_c - h

max_p_index = top_one_third_periods.idxmax()
max_p_wave = top_one_third_waves[max_p_index]

s_op = 2 * np.pi * max_p_wave / (9.81 * max(top_one_third_periods) ** 2)

b = -5.42 * s_op + 0.0323 * (Hs/Dn_50) - 0.0017 * (B/Dn_50) ** (1.84) + 0.51

Rc_range = list(np.arange(-2, 3, 0.5))
Ct_range = []

for Rc in Rc_range:
    Ct = (0.031 * (Hs/Dn_50) - 0.24) * (Rc/Dn_50) + b
    Ct_range.append(Ct)

Ct = (0.031 * (Hs/Dn_50) - 0.24) * (R_c/Dn_50) + b # Van der meer and d'angermond

# S_op, Hs/Dn50 and Rc/Dn50 are satisfied

# plt.figure(figsize=(10, 6))
# plt.plot(Rc_range/Dn_50, Ct_range, marker='x', color='blue')
# plt.xlabel('Rc/D50')
# plt.ylabel('Ct')
# plt.title('Wave Transmission')
# plt.show()

# print(Dn_50)
# print(str(B/Dn_50))
# print(str(Hs/Dn_50))
# print(str(s_op))
# print(str(Ct))
