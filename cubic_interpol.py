import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define the interval and the function
interval_start = 0
interval_end = 9*np.pi
num_samples = 3
num_interp_points = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points
yi = np.sin(xi)

# Create the interpolation function
cubic_interp = CubicSpline(xi, yi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Evaluate the interpolated function at desired points
interpolated_values = cubic_interp(x_values)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.sin(x_values), 'g-', label='sin(x)')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated')
plt.legend()
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Interpolation of sin(x) over [0, π]')
plt.show()


# Define the interval and the function (e^x)
interval_start = 0
interval_end = 2
num_samples = 3
num_interp_points = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points (e^x)
yi = np.exp(xi)

# Create the cubic spline interpolator
cubic_interp = CubicSpline(xi, yi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Evaluate the cubic spline at desired points
interpolated_values = cubic_interp(x_values)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.exp(x_values), 'g-', label='e^x')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Cubic Spline)')
plt.legend()
plt.xlabel('x')
plt.ylabel('e^x')
plt.title('Cubic Spline Interpolation of e^x')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Define the interval and the function (cosh(x))
interval_start = -2
interval_end = 2
num_samples = 3
num_interp_points = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points (cosh(x))
yi = np.cosh(xi)

# Create the cubic spline interpolator
cubic_interp = CubicSpline(xi, yi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Evaluate the cubic spline at desired points
interpolated_values = cubic_interp(x_values)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.cosh(x_values), 'g-', label='cosh(x)')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Cubic Spline)')
plt.legend()
plt.xlabel('x')
plt.ylabel('cosh(x)')
plt.title('Cubic Spline Interpolation of cosh(x)')
plt.show()
