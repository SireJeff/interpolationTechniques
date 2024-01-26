import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# Define the polar function (r = e^θ)
theta = np.linspace(0, 2*np.pi, 9)
r = np.exp(theta)

# Convert polar coordinates to complex numbers
z = r * np.exp(1j * theta)

# Create the PchipInterpolator for complex numbers
pchip_interp_z = PchipInterpolator(theta, z, axis=0, extrapolate='periodic')

# Define points at which you want to evaluate the interpolated function
theta_interp = np.linspace(0, 2*np.pi, 100)

# Evaluate the PchipInterpolator at desired points
z_interp = pchip_interp_z(theta_interp)

# Convert complex numbers back to polar coordinates for plotting
r_interp = np.abs(z_interp)
theta_interp_values = np.angle(z_interp)

# Plot the original polar function and the interpolated values
plt.polar(theta, r, 'bo', label='Sample Points')
plt.plot(theta_interp_values, r_interp, 'r--', label='Interpolated')
plt.show()

# Plot the Cartesian coordinates for visualization
plt.plot(r_interp * np.cos(theta_interp_values), r_interp * np.sin(theta_interp_values), 'r--', label='Interpolated (Cartesian)')
plt.scatter(r * np.cos(theta), r * np.sin(theta), color='blue')
plt.axis('equal')
plt.show()


