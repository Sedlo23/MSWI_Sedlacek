import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

# Definice funkcí
def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def harmonic(x, A, omega, phi):
    return A * np.sin(omega * x + phi)

def logarithm(x, a, b):
    return a * np.log(x) + b

# Generování náhodných dat a přidání šumu
x = np.linspace(1, 10, 50)
y_p = polynomial(x, 1, -2, 3) + np.random.normal(0, 0.5, 50)
y_h = harmonic(x, 1, 2*np.pi, np.pi/4) + np.random.normal(0, 0.2, 50)
y_l = logarithm(x, 1, 1) + np.random.normal(0, 0.1, 50)

# Interpolace pomocí lineární regrese
lr_p = LinearRegression().fit(x.reshape(-1, 1), y_p)
lr_h = LinearRegression().fit(x.reshape(-1, 1), y_h)
lr_l = LinearRegression().fit(np.log(x).reshape(-1, 1), y_l)

# Interpolace pomocí kvadratického polynomu
popt_p, _ = curve_fit(polynomial, x, y_p)
popt_h, _ = curve_fit(harmonic, x, y_h)
popt_l, _ = curve_fit(logarithm, x, y_l)

# Interpolace pomocí lineárního spline
spline_p = interp1d(x, y_p, kind="linear")
spline_h = interp1d(x, y_h, kind="linear")
spline_l = interp1d(x, y_l, kind="linear")

# Výpočet součtu čtverců chyb pro každou metodu
p_err_lr = np.sum((y_p - lr_p.predict(x.reshape(-1, 1)))**2)
p_err_qp = np.sum((y_p - polynomial(x, *popt_p))**2)
p_err_ls = np.sum((y_p - spline_p(x))**2)

h_err_lr = np.sum((y_h - lr_h.predict(x.reshape(-1, 1)))**2)
h_err_hf = np.sum((y_h - harmonic(x, *popt_h))**2)
h_err_ls = np.sum((y_h - spline_h(x))**2)

l_err_lr = np.sum((y_l - lr_l.predict(np.log(x).reshape(-1, 1)))**2)
l_err_lf = np.sum((y_l - logarithm(x, *popt_l))**2)
l_err_ls = np.sum((y_l - spline_l(x))**2)

# Vykreslení grafů
fig, ax = plt.subplots(3, 1, figsize=(8, 12))

ax[0].scatter(x, y_p, label="Data")
ax[0].plot(x, lr_p.predict(x.reshape(-1, 1)), label="Linear Regression: {:.2f}".format(p_err_lr))
ax[0].plot(x, polynomial(x, *popt_p), label="Quadratic Polynomial: {:.2f}".format(p_err_qp))
ax[0].plot(x, spline_p(x), label="Linear Spline: {:.2f}".format(p_err_ls))
ax[0].set_xlabel("x")
ax[0].set_ylabel("y")
ax[0].set_title("Polynomial Interpolation")
ax[0].legend()

ax[1].scatter(x, y_h, label="Data")
ax[1].plot(x, lr_h.predict(x.reshape(-1, 1)), label="Linear Regression: {:.2f}".format(h_err_lr))
ax[1].plot(x, harmonic(x, *popt_h), label="Harmonic Function: {:.2f}".format(h_err_hf))
ax[1].plot(x, spline_h(x), label="Linear Spline: {:.2f}".format(h_err_ls))
ax[1].set_xlabel("x")
ax[1].set_ylabel("y")
ax[1].set_title("Harmonic Interpolation")
ax[1].legend()

ax[2].scatter(x, y_l, label="Data")
ax[2].plot(x, lr_l.predict(np.log(x).reshape(-1, 1)), label="Linear Regression: {:.2f}".format(l_err_lr))
ax[2].plot(x, logarithm(x, *popt_l), label="Logarithmic Function: {:.2f}".format(l_err_lf))
ax[2].plot(x, spline_l(x), label="Linear Spline: {:.2f}".format(l_err_ls))
ax[2].set_xlabel("x")
ax[2].set_ylabel("y")
ax[2].set_title("Logarithmic Interpolation")
ax[2].legend()

plt.show()
