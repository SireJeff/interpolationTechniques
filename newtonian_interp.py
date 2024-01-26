import numpy as np
import matplotlib.pyplot as plt

def divided_diff(x, y):
    n = len(x)
    F = np.zeros((n, n))

    # Set the first column of F to y
    F[:, 0] = y

    # Compute the divided differences
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i + 1, j - 1] - F[i, j - 1]) / (x[i + j] - x[i])

    return F[0, :]

def newton_interpolation(x, y, x_interp):
    n = len(x)
    F = divided_diff(x, y)
    result = F[0]

    for i in range(1, n):
        term = F[i]
        for j in range(i):
            term *= (x_interp - x[j])
        result += term

    return result

# Define the interval and the function (cosh(x))
interval_start = -2
interval_end = 2
num_samples = 3
num_interp_points = 100

# Generate sample points within the interval
xi = np.linspace(interval_start, interval_end, num_samples)

# Compute the function values at these sample points (cosh(x))
yi = np.cosh(xi)

# Define points at which you want to evaluate the interpolated function
x_values = np.linspace(interval_start, interval_end, num_interp_points)

# Perform Newtonian interpolation
interpolated_values = [newton_interpolation(xi, yi, x) for x in x_values]

# Plot the original function and the interpolated values
plt.plot(xi, yi, 'bo', label='Sample Points')
plt.plot(x_values, np.cosh(x_values), 'g-', label='cosh(x)')
plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Newtonian)')
plt.legend()
plt.xlabel('x')
plt.ylabel('cosh(x)')
plt.title('Newtonian Interpolation of cosh(x)')
plt.show()
