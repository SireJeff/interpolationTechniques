import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

def polar_interpolation(theta, r, num_interp_points=100):
    z = r * np.exp(1j * theta)
    pchip_interp_z = PchipInterpolator(theta, z, axis=0, extrapolate='periodic')

    theta_interp = np.linspace(0, 2*np.pi, num_interp_points)
    z_interp = pchip_interp_z(theta_interp)

    r_interp = np.abs(z_interp)
    theta_interp_values = np.angle(z_interp)

    return theta_interp, r_interp, theta_interp_values

def plot_polar_interpolation(theta, r, num_interp_points=100):
    theta_interp, r_interp, theta_interp_values = polar_interpolation(theta, r, num_interp_points)

    # Plot the original polar function and the interpolated values
    plt.polar(theta, r, 'bo', label='Sample Points')
    plt.plot(theta_interp_values, r_interp, 'r--', label='Interpolated')
    plt.show()

    # Plot the Cartesian coordinates for visualization
    plt.plot(r_interp * np.cos(theta_interp_values), r_interp * np.sin(theta_interp_values), 'r--', label='Interpolated (Cartesian)')
    plt.scatter(r * np.cos(theta), r * np.sin(theta), color='blue')
    plt.axis('equal')
    plt.show()

# Example usage:
theta = np.linspace(0, 2*np.pi, 9)
r = np.exp(theta)
plot_polar_interpolation(theta, r)
