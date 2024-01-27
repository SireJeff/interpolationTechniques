import numpy as np
import matplotlib.pyplot as plt

def divided_diff(x, y):
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y

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

def perform_newton_interpolation(interval_start, interval_end, num_samples, num_interp_points, func, func_name):
    xi = np.linspace(interval_start, interval_end, num_samples)
    yi = func(xi)
    
    x_values = np.linspace(interval_start, interval_end, num_interp_points)
    interpolated_values = [newton_interpolation(xi, yi, x) for x in x_values]

    plt.plot(xi, yi, 'bo', label='Sample Points')
    plt.plot(x_values, func(x_values), 'g-', label=f'{func_name}(x)')
    plt.plot(x_values, interpolated_values, 'r--', label='Interpolated (Newtonian)')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel(f'{func_name}(x)')
    plt.title(f'Newtonian Interpolation of {func_name}(x)')
    plt.show()

# # Example 1: cosh(x)
# perform_newton_interpolation(-2, 2, 3, 100, np.cosh, 'cosh')

# # Example 2: sin(x)
# perform_newton_interpolation(-np.pi, 9*np.pi,18, 100, np.sin, 'sin')

# # Example 3: e^x
# perform_newton_interpolation(-2, 2, 3, 100, np.exp, 'e^x')
