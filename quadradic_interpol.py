import numpy as np
import matplotlib.pyplot as plt

def quadratic_spline(xi, yi):
    n = len(xi) - 1  # Number of intervals
    a = yi.copy()
    b = np.zeros(n)
    c = np.zeros(n)

    for i in range(n):
        h = xi[i + 1] - xi[i]
        if i < n - 1:
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

def perform_quadratic_spline_interpolation(interval_start, interval_end, num_samples, num_interp_points, func, func_name):
    xi = np.linspace(interval_start, interval_end, num_samples)
    yi = func(xi)
    
    a, b, c = quadratic_spline(xi, yi)
    
    x_values = np.linspace(interval_start, interval_end, num_interp_points)
    interpolated_values = evaluate_quadratic_spline(x_values, xi, a, b, c)

    plt.plot(xi, yi, 'bo', label='Sample Points')
    plt.plot(x_values, func(x_values), 'g-', label=f'{func_name}(x)')
    plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Quadratic Spline)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel(f'{func_name}(x)')
    plt.title(f'Quadratic Spline Interpolation of {func_name}(x)')
    plt.show()

# # Example 1: sin(x)
# perform_quadratic_spline_interpolation(-np.pi, 9 * np.pi, 18, 100, np.sin, 'sin')

# # Example 2: e^x
# perform_quadratic_spline_interpolation(-2, 2, 3, 100, np.exp, 'e^x')

# # Example 3: cosh(x)
# perform_quadratic_spline_interpolation(-2, 2, 3, 100, np.cosh, 'cosh')
