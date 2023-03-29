import numpy as np

def adaptive_deriv(f, x, h=1e-5, tol=1e-6, max_iter=10):
    """Numerická derivace funkce f s adaptabilním krokem"""
    y0 = f(x)
    h0 = h
    for i in range(max_iter):
        y1 = f(x+h)
        y2 = f(x+2*h)
        dy1 = (y1 - y0) / h
        dy2 = (y2 - y1) / h
        err = np.abs(dy2 - dy1) / np.max(np.abs([y2, y1]))
        if err < tol:
            return dy2
        h = 0.9 * h * (tol / err) ** 0.2
    raise ValueError("Adaptive derivative did not converge after %d iterations" % max_iter)

def pow(x):
    return x*x

print(adaptive_deriv(pow,5))

print(adaptive_deriv(pow,10))

print(adaptive_deriv(pow,50))