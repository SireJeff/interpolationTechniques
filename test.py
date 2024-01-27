import numpy as np
import matplotlib.pyplot as plt
from compare_performance import compare_performance
from quadradic_interpol import quadratic_spline,evaluate_quadratic_spline,perform_quadratic_spline_interpolation
from cubic_interpol import cubic_spline_interpolation , generate_sample_points
from newtonian_interpol import newton_interpolation,perform_newton_interpolation, divided_diff

# Define the interval and the function
interval_start = 0
interval_end = 9 * np.pi
num_samples = 20

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points
yi = np.sin(xi)

# Test the performance of interpolation methods
cubic_interp, a, b, c, x_values, newtonian_values, cubic_time, quadratic_time, newtonian_time = compare_performance(xi, yi)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.sin(x_values), 'g-', label='sin(x)')
plt.plot(x_values, cubic_interp(x_values), 'r--', label='Cubic Spline')
plt.plot(x_values, evaluate_quadratic_spline(x_values, xi, a, b, c), 'm--', label='Quadratic Spline')
plt.plot(x_values, newtonian_values, 'c--', label='Newtonian Interpolation')
plt.legend()
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Interpolation Comparison for sin(x)')
plt.show()

# Print the time taken by each method
print(f'Cubic Spline Time: {cubic_time} seconds')
print(f'Quadratic Spline Time: {quadratic_time} seconds')
print(f'Newtonian Interpolation Time: {newtonian_time} seconds')
