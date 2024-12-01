import numpy as np
import matplotlib.pyplot as plt

# Given D_85
D_85 = 170  # Example value in mm

# Calculate k using D_85
k = -np.log(1 - 0.85) / D_85

# Define percentiles and corresponding values
percentiles = [0.15, 0.5, 0.85]
D_values = [-np.log(1 - F) / k for F in percentiles]  # Calculate D15, D50, D85

# Generate the distribution
d = np.linspace(0, 175, 500)  # Grain sizes to plot
F_d = 1 - np.exp(-k * d)  # Cumulative distribution function

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(d, F_d, label="Grain Size Distribution", color='blue')
plt.scatter(D_values, percentiles, color='red', label="D values (D15, D50, D85)")
plt.axhline(0.85, color='gray', linestyle='--', label="D85 = {:.2f} mm".format(D_85))
plt.axhline(0.5, color='gray', linestyle='--', label="D50 = {:.2f} mm".format(D_values[1]))
plt.axhline(0.15, color='gray', linestyle='--', label="D15 = {:.2f} mm".format(D_values[0]))

# Annotations
for i, (D, F) in enumerate(zip(D_values, percentiles)):
    plt.annotate(f"D{int(F*100)} = {D:.2f} mm", xy=(D, F), xytext=(D + 0.1, F - 0.05),
                 arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=10)

# Labels and Legend
plt.title("Grain Size Distribution (Exponential)")
plt.xlabel("Grain Size (mm)")
plt.ylabel("Cumulative Percent Finer")
plt.legend()
plt.grid(True)
plt.show()
