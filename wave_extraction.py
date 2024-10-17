import pandas as pd
import numpy as np



data_path = 'project_data/'

dates = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

for date in dates:

    # Load only the relevant columns: date/time and wave height (WVHT)
    df = pd.read_csv(data_path + '44095h' + str(date) + '.txt', sep=r'\s+', header=0, skiprows=[1], usecols=['WVHT'])

    df = df.dropna(subset=['WVHT'])
    df_filtered = df[df['WVHT'] != 99]

    # Sort WVHT in descending order
    df_sorted = df_filtered['WVHT'].sort_values(ascending=False)
    top_one_third_count = int(len(df_sorted) / 3)
    top_one_third_waves = df_sorted[:top_one_third_count]


    average_wvht = df_filtered['WVHT'].mean()
    rms_wvht = np.sqrt(np.mean(df_filtered['WVHT']**2))
    average_top_one_third = top_one_third_waves.mean()
    max_wvht = df_filtered['WVHT'].max()

    # Display the average
    print("Data for the year " + str(date))
    print(f"Average Wave Height (WVHT): {average_wvht:.2f}")
    print(f"Root Mean Squared Wave Height (WVHT): {rms_wvht:.2f}")
    print(f"Significant Wave Height (WVHT): {average_top_one_third:.2f}")
    print(f"Maximum Wave Height (WVHT): {max_wvht:.2f}")
    print(" ")

