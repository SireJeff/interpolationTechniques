import numpy as np
from scipy.interpolate import CubicSpline
from timeit import default_timer as timer
from quadradic_interpol import quadratic_spline,evaluate_quadratic_spline,perform_quadratic_spline_interpolation
from cubic_interpol import cubic_spline_interpolation , generate_sample_points
from newtonian_interpol import newton_interpolation,perform_newton_interpolation,divided_diff

def cubic_spline_interpolation(x, y):
    cubic_interp = CubicSpline(x, y)
    return cubic_interp

def quadratic_spline_interpolation(xi, yi):
    a, b, c = quadratic_spline(xi, yi)
    return a, b, c

def newtonian_interpolation(x, y, x_interp):
    F = divided_diff(x, y)
    result = F[0]

    for i in range(1, len(x)):
        term = F[i]
        for j in range(i):
            term *= (x_interp - x[j])
        result += term

    return result

def compare_performance(x, y):
    # Measure time for cubic spline interpolation
    start_time = timer()
    cubic_interp = cubic_spline_interpolation(x, y)
    cubic_time = timer() - start_time

    # Measure time for quadratic spline interpolation
    start_time = timer()
    a, b, c = quadratic_spline_interpolation(x, y)
    quadratic_time = timer() - start_time

    # Measure time for Newtonian interpolation
    start_time = timer()
    x_values = np.linspace(min(x), max(x), 100)
    newtonian_values = [newtonian_interpolation(x, y, xi) for xi in x_values]
    newtonian_time = timer() - start_time

    return cubic_interp, a, b, c, x_values, newtonian_values, cubic_time, quadratic_time, newtonian_time
