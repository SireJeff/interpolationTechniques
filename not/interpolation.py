import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, RectBivariateSpline
from timeit import default_timer as timer
from scipy.interpolate import interp2d
# Function to perform spline interpolation
def spline_interpolation(x, y):
    # Cubic spline interpolation
    cs = CubicSpline(x, y)
    return cs


# Function to perform bicubic interpolation using spline interpolation result
def bicubic_interp(cs, x, y):
    # Bicubic interpolation
    x_values = np.linspace(min(x), max(x), 100)
    y_values = np.linspace(min(y), max(y), 100)
    X, Y = np.meshgrid(x_values, y_values)
    Z = cs(X[0])
    
    bicubic_interp = interp2d(x_values, y_values, Z, kind='cubic')
    return bicubic_interp, x_values, y_values




def compare_performance(x, y):
    # Measure time for spline interpolation
    start_time = timer()
    cs = spline_interpolation(x, y)
    spline_time = timer() - start_time

    # Measure time for bicubic interpolation
    start_time = timer()
    bicubic_interp_r, x_values, y_values = bicubic_interp(cs, x, y)
    bicubic_time = timer() - start_time

    # Measure time for other interpolation methods (e.g., Newton's divided differences, Lagrange)
    # ... (implement other interpolation methods here)

    # return cs, bicubic_interp_r, x_values, y_values, spline_time, bicubic_time


    return cs, bicubic_interp_r, spline_time, bicubic_time

# Function to visualize interpolation results
def visualize_interpolation(x, y, cs, bicubic_interp_r, other_interpolations=None):
    # Plot original data points
    plt.scatter(x, y, label='Original Data Points')

    # Plot spline interpolation
    x_new = np.linspace(min(x), max(x), 1000)
    y_spline = cs(x_new)
    plt.plot(x_new, y_spline, label='Spline Interpolation', linestyle='--')

    # Plot bicubic interpolation
    Z = cs(x_new)
    bicubic_values = bicubic_interp_r(x_new, y)
    plt.plot(x_new, bicubic_values, label='Bicubic Interpolation', linestyle='--')

    # Plot other interpolation methods
    # ... (implement other interpolation methods here)

    plt.legend()
    plt.title('Comparison of Interpolation Methods')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Example data points
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 4, 1, 5, 3])

# Compare performance and get interpolation results
cs, bicubic_interp_r, spline_time, bicubic_time = compare_performance(x_data, y_data)

# Print performance results
print(f"Spline Interpolation Time: {spline_time} seconds")
print(f"Bicubic Interpolation Time: {bicubic_time} seconds")

# Visualize interpolation results
visualize_interpolation(x_data, y_data, cs, bicubic_interp_r)
