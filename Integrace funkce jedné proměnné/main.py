import numpy as np
import matplotlib.pyplot as plt

def euler_step(f, x, dt):
    return x + f(x) * dt

def lorenz_system(x):
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])

dt = 0.01
T = 50.0
t = np.arange(0.0, T, dt)

x = np.zeros((len(t), 3))
x[0, :] = [0.0, 1.0, 0.0]

for i in range(len(t) - 1):
    x[i + 1, :] = euler_step(lorenz_system, x[i, :], dt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x[:, 0], x[:, 1], x[:, 2], color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz System')
plt.show()


