import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh



data_path = 'project_data/'

use_situ = True

dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

all_wave_heights = []

if use_situ == True:

    for date in dates:
        df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['WVHT'])
        df = df.dropna(subset=['WVHT'])
        df_filtered = df[df['WVHT'] != 99]
        all_wave_heights.extend(df_filtered['WVHT'].tolist())

    all_wave_heights = pd.Series(all_wave_heights)

elif use_situ == False:

    df = pd.read_csv("project_data/off_sig_h.csv", skiprows=7)

    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['VHM0_WW'])
    df_filtered = df[df['VHM0_WW'] != 0]
    all_wave_heights.extend(df_filtered['VHM0_WW'].tolist())

    all_wave_heights = pd.Series(all_wave_heights)

# Sort the data to calculate the top one-third of wave heights
all_wave_heights_sorted = all_wave_heights.sort_values(ascending=False)
top_one_third_count = int(len(all_wave_heights_sorted) / 3)
top_one_third_waves = all_wave_heights_sorted[:top_one_third_count]

# Calculate the overall wave stats
avg_wvht = all_wave_heights.mean()
rms_wvht = np.sqrt(np.mean(all_wave_heights**2))
significant_wvht = top_one_third_waves.mean()
max_wvht = all_wave_heights.max()


# Plot the histogram and rayleigh for wave heights
plt.figure(figsize=(8, 6))
count, bins, ignored = plt.hist(all_wave_heights, bins=30, density=True, alpha=1, color='lightblue', edgecolor='blue', linewidth= 1, linestyle = ':' )
param = rayleigh.fit(all_wave_heights)
x = np.linspace(0, max(all_wave_heights), 100)
pdf_fitted = rayleigh.pdf(x, *param)
plt.plot(x, pdf_fitted, 'r-', lw=2, label='Rayleigh PDF')

# Add labels and title
plt.xlabel('Wave Height (meters)')
plt.ylabel('Density')
plt.title('Histogram and Rayleigh PDF of Wave Heights')

# Display the PDF and histogram plot
plt.legend()
plt.show()
