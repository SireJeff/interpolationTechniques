import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_sample_points(interval_start, interval_end, num_samples, func):
    xi = np.linspace(interval_start, interval_end, num_samples)
    yi = func(xi)
    return xi, yi

def cubic_spline_interpolation(xi, yi, interval_start, interval_end, num_interp_points, func):
    cubic_interp = CubicSpline(xi, yi)
    
    x_values = np.linspace(interval_start, interval_end, num_interp_points)
    interpolated_values = cubic_interp(x_values)
    
    plt.plot(xi, yi, 'bo', label='Sample Points')
    plt.plot(x_values, func(x_values), 'g-', label=f'{func.__name__}(x)')
    plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Cubic Spline)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel(f'{func.__name__}(x)')
    plt.title(f'Cubic Spline Interpolation of {func.__name__}(x)')
    plt.show()

# # Example 1: sin(x)
# interval_start = -np.pi
# interval_end = 9*np.pi
# num_samples = 18
# num_interp_points = 100

# xi, yi = generate_sample_points(interval_start, interval_end, num_samples, np.sin)
# cubic_spline_interpolation(xi, yi, interval_start, interval_end, num_interp_points, np.sin)

# # Example 2: e^x
# interval_start = -2
# interval_end = 2
# num_samples = 3
# num_interp_points = 100

# xi, yi = generate_sample_points(interval_start, interval_end, num_samples, np.exp)
# cubic_spline_interpolation(xi, yi, interval_start, interval_end, num_interp_points, np.exp)

# # Example 3: cosh(x)
# interval_start = -2
# interval_end = 2
# num_samples = 3
# num_interp_points = 100

# xi, yi = generate_sample_points(interval_start, interval_end, num_samples, np.cosh)
# cubic_spline_interpolation(xi, yi, interval_start, interval_end, num_interp_points, np.cosh)
