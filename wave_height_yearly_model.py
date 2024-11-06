import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file, skipping initial metadata rows
df = pd.read_csv("project_data/off_sig_h.csv", skiprows=7)
df.columns = df.columns.str.strip()  # Ensure no extra spaces in column names

# Extract the year from the 'time' column
df['year'] = df['time'].str[:4].astype(int)  # Convert year to integer

# List of unique years in the dataset
dates = sorted(df['year'].unique())

# Lists to store the results for each year
average_wvht_list = []
rms_wvht_list = []
average_top_one_third_list = []
max_wvht_list = []

for date in dates:
    # Filter data for the current year
    df_year = df[df['year'] == date].dropna(subset=['VHM0_WW'])
    df_filtered = df_year[df_year['VHM0_WW'] != 99]

    # Sort VHM0_WW in descending order and select the top 1/3
    df_sorted = df_filtered['VHM0_WW'].sort_values(ascending=False)
    top_one_third_count = int(len(df_sorted) / 3)
    top_one_third_waves = df_sorted[:top_one_third_count]

    # Calculate statistics for the year
    average_wvht = df_filtered['VHM0_WW'].mean()
    rms_wvht = np.sqrt(np.mean(df_filtered['VHM0_WW']**2))
    average_top_one_third = top_one_third_waves.mean()
    max_wvht = df_filtered['VHM0_WW'].max()

    # Append results to lists
    average_wvht_list.append(average_wvht)
    rms_wvht_list.append(rms_wvht)
    average_top_one_third_list.append(average_top_one_third)
    max_wvht_list.append(max_wvht)

    # Print year-specific results
    print(f"Data for the year {date}")
    print(f"Average Wave Height (H_z): {average_wvht:.2f}")
    print(f"Root Mean Squared Wave Height (H_rms): {rms_wvht:.2f}")
    print(f"Significant Wave Height (H_s): {average_top_one_third:.2f}")
    print(f"Maximum Wave Height (H_max): {max_wvht:.2f}")
    print(" ")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dates, average_wvht_list, marker='o', label='Average Wave Height (H_z)', color='blue')
plt.plot(dates, rms_wvht_list, marker='x', label='RMS Wave Height (H_rms)', color='green')
plt.plot(dates, average_top_one_third_list, marker='s', label='Significant Wave Height (H_s)', color='red')
plt.plot(dates, max_wvht_list, marker='^', label='Maximum Wave Height (H_max)', color='purple')

plt.xlabel('Year')
plt.ylabel('Wave Height (meters)')
plt.title('Wave Height Statistics Over the Years')
plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.8))
plt.grid(True)
plt.xticks(dates, rotation=45)

plt.show()
