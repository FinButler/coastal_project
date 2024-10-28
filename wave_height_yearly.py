import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data path
data_path = 'project_data/'

# List of years
dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

# Lists to store the results for each year
average_wvht_list = []
rms_wvht_list = []
average_top_one_third_list = []
max_wvht_list = []

for date in dates:

    df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['WVHT'])
    df = df.dropna(subset=['WVHT'])
    df_filtered = df[df['WVHT'] != 99]

    # Sort WVHT in descending order and select the top 1/3
    df_sorted = df_filtered['WVHT'].sort_values(ascending=False)
    top_one_third_count = int(len(df_sorted) / 3)
    top_one_third_waves = df_sorted[:top_one_third_count]

    # Calculate statistics for each year
    average_wvht = df_filtered['WVHT'].mean()
    rms_wvht = np.sqrt(np.mean(df_filtered['WVHT']**2))
    average_top_one_third = top_one_third_waves.mean()
    max_wvht = df_filtered['WVHT'].max()

    # Append the results to the lists
    average_wvht_list.append(average_wvht)
    rms_wvht_list.append(rms_wvht)
    average_top_one_third_list.append(average_top_one_third)
    max_wvht_list.append(max_wvht)


    print(f"Data for the year {date}")
    print(f"Average Wave Height (H_z): {average_wvht:.2f}")
    print(f"Root Mean Squared Wave Height (H_rms): {rms_wvht:.2f}")
    print(f"Significant Wave Height (H_s): {average_top_one_third:.2f}")
    print(f"Maximum Wave Height (H_max): {max_wvht:.2f}")
    print(" ")


plt.figure(figsize=(10, 6))

plt.plot(dates, average_wvht_list, marker='o', label='Average Wave Height (H_z)', color='blue')
plt.plot(dates, rms_wvht_list, marker='x', label='RMS Wave Height (H_rms)', color='green')
plt.plot(dates, average_top_one_third_list, marker='s', label='Significant Wave Height (H_s)', color='red')
plt.plot(dates, max_wvht_list, marker='^', label='Maximum Wave Height (H_max)', color='purple')

plt.xlabel('Year')
plt.ylabel('Wave Height (meters)')
plt.title('Wave Height Statistics Over the Years')
plt.legend()
plt.grid(True)

plt.show()
