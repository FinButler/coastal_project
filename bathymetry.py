import xarray as xr

# # Open the NetCDF file
# file_path = "project_data/gebco_2024_n35.61_s35.6_w-75.47_e-75.33.nc"  # Replace with your file's path
# ds = xr.open_dataset(file_path)
#
# # Display the dataset structure
# print(ds)
#
# # List all variables
# print("\nVariables:")
# print(ds.data_vars)
#
# # Access global attributes
# print("\nGlobal Attributes:")
# print(ds.attrs)
#
# # Inspect a variable
# var_name = "elevation"  # Replace with the variable you're interested in
# if var_name in ds:
#     var = ds[var_name]
#     print(f"\nDetails of variable '{var_name}':")
#     print(var)
#
#     # Access the data
#     print("\nData:")
#     print(var.values)
# else:
#     print(f"Variable '{var_name}' not found!")
#
# # Close the dataset
# ds.close()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Given data
transect_distance = np.arange(0, 14750 + 450, 450)  # Distances in meters (0 to 14,750, step of 450 m)
depths = [0, 4, 9, 9, 9, 10, 11, 12, 12, 13, 14, 15, 14, 15, 15, 15, 14, 14,
          16, 16, 17, 16, 16, 16, 19, 19, 19, 18, 21, 24, 26, 29, 29, 30]  # Depth values

# Generate distances for interpolation (1m resolution)
interp_distances = np.arange(0, 14750)  # 14,750 entries (from 0 to 14,749 meters)

# Perform Cubic Spline interpolation
cubic_spline_interpolator = CubicSpline(transect_distance, depths)
interpolated_depths_cubic = cubic_spline_interpolator(interp_distances)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(transect_distance, depths, 'o', label='Original Data (450m intervals)', markersize=5, color='black')
plt.plot(interp_distances, interpolated_depths_cubic, '-', label='Cubic Spline Interpolation', alpha=0.7)
plt.xlabel('Distance along Transect (m)')
plt.ylabel('Depth (m)')
plt.title('Cubic Spline Interpolation of Depth Along Transect')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Optionally, save the plot
# plt.savefig('cubic_spline_interpolation.png', dpi=300)

# Save the interpolated data (Optional)
output_data_cubic = np.column_stack((interp_distances, interpolated_depths_cubic))
np.savetxt("project_data/transect_cubic_spline_14750_entries.csv", output_data_cubic, delimiter=",", header="Distance (m),Depth (m)", comments="")
