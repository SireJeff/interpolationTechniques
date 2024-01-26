import numpy as np
import matplotlib.pyplot as plt

def quadratic_spline(xi, yi):
    n = len(xi) - 1  # Number of intervals

    # Initialize arrays to store coefficients
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)

    # Calculate coefficients for each quadratic polynomial
    for i in range(n):
        h = xi[i + 1] - xi[i]
        a[i] = yi[i]
        if i < n - 1:  # Check if we are not at the last interval
            b[i] = (yi[i + 1] - yi[i]) / h - h * (2 * c[i] + c[i + 1]) / 3.0
        c[i] = (yi[i + 1] - yi[i]) / h

    return a, b, c

def evaluate_quadratic_spline(x, xi, a, b, c):
    n = len(xi) - 1
    y = np.zeros_like(x)

    for i in range(n):
        mask = np.logical_and(xi[i] <= x, x <= xi[i + 1])
        h = x[mask] - xi[i]
        y[mask] = a[i] + b[i] * h + c[i] * h**2

    return y

# Define the interval and the function
interval_start = 0
interval_end = 9 * np.pi
num_samples = 3
num_interp_points = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points
yi = np.sin(xi)

# Create the quadratic spline coefficients
a, b, c = quadratic_spline(xi, yi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Evaluate the quadratic spline at desired points
interpolated_values = evaluate_quadratic_spline(x_values, xi, a, b, c)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.sin(x_values), 'g-', label='sin(x)')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Quadratic Spline)')
plt.legend()
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Quadratic Spline Interpolation of sin(x)')
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

# Create the quadratic spline coefficients
a, b, c = quadratic_spline(xi, yi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Evaluate the quadratic spline at desired points
interpolated_values = evaluate_quadratic_spline(x_values, xi, a, b, c)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.exp(x_values), 'g-', label='e^x')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Quadratic Spline)')
plt.legend()
plt.xlabel('x')
plt.ylabel('e^x')
plt.title('Quadratic Spline Interpolation of e^x')
plt.show()

# Define the interval and the function (cosh(x))
interval_start = -2
interval_end = 2
num_samples = 3
num_interp_points = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points (cosh(x))
yi = np.cosh(xi)

# Create the quadratic spline coefficients
a, b, c = quadratic_spline(xi, yi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Evaluate the quadratic spline at desired points
interpolated_values = evaluate_quadratic_spline(x_values, xi, a, b, c)

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.cosh(x_values), 'g-', label='cosh(x)')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Quadratic Spline)')
plt.legend()
plt.xlabel('x')
plt.ylabel('cosh(x)')
plt.title('Quadratic Spline Interpolation of cosh(x)')
plt.show()