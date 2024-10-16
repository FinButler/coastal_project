import pandas as pd
import matplotlib.pyplot as plt

data_path = 'project_data/'

# Load only the relevant columns: date/time and wave height (WVHT)
df = pd.read_csv(data_path + '44095h2012.txt', sep=r'\s+', header=0, skiprows=[1],
                 usecols=['#YY', 'MM', 'DD', 'hh', 'mm', 'WVHT'])

# Check the DataFrame structure
print(df.head())  # Inspect the first few rows

# Convert date and time columns to string for plotting
df['Date'] = df['#YY'].astype(str) + '-' + df['MM'].astype(str).str.zfill(2) + '-' + df['DD'].astype(str).str.zfill(2)
df['Time'] = df['hh'].astype(str).str.zfill(2) + ':' + df['mm'].astype(str).str.zfill(2)

# Create a new DataFrame for plotting
plot_df = df[['Date', 'Time', 'WVHT']]

# Plot the wave height against the date and time separately
plt.figure(figsize=(10, 6))

# Plot the wave height with markers for each timestamp
for index, row in plot_df.iterrows():
    plt.scatter(row['Date'] + ' ' + row['Time'], row['WVHT'], color='b', label='Wave Height' if index == 0 else "")

plt.xlabel('Date and Time')
plt.ylabel('Wave Height (m)')
plt.title('Wave Height over Time')
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels

# Show the plot
plt.show()
