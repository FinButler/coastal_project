import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh

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
avg_wvht = all_wave_heights.mean()
rms_wvht = np.sqrt(np.mean(all_wave_heights**2))
significant_wvht = top_one_third_waves.mean()
max_wvht = all_wave_heights.max()

# Plot the horizontal bar chart for wave height statistics
plt.figure(figsize=(8, 6))
bars = plt.barh(['Average Wave Height (H_z)', 'RMS Wave Height (H_rms)',
                 'Significant Wave Height (H_s)', 'Maximum Wave Height (H_max)'],
                [avg_wvht, rms_wvht, significant_wvht, max_wvht],
                color=['blue', 'green', 'red', 'purple'])

# Add values to each bar
for bar in bars:
    plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}', va='center')

# Labels and title for bar chart
plt.xlabel('Wave Height (meters)')
plt.title('Wave Height Statistics Averaged Over All Years')

# Display the bar chart
plt.show()

# Now plot the histogram and Rayleigh PDF overlay

# Plot the histogram of wave heights
plt.figure(figsize=(8, 6))
count, bins, ignored = plt.hist(all_wave_heights, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')

# Fit a Rayleigh distribution to the data
param = rayleigh.fit(all_wave_heights)  # Fit returns the parameters for the Rayleigh distribution

# Plot the Rayleigh PDF
x = np.linspace(0, max(all_wave_heights), 100)
pdf_fitted = rayleigh.pdf(x, *param)  # Generate the PDF using the fitted parameters
plt.plot(x, pdf_fitted, 'r-', lw=2, label='Rayleigh PDF')

# Add labels and title
plt.xlabel('Wave Height (meters)')
plt.ylabel('Density')
plt.title('Histogram and Rayleigh PDF of Wave Heights')

# Display the PDF and histogram plot
plt.legend()
plt.show()
