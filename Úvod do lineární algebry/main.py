import time
import numpy as np
import matplotlib.pyplot as plt

# Funkce pro generování náhodné matice
def generate_matrix(n):
    return np.random.rand(n, n)


# Funkce pro generování náhodné pravé strany
def generate_b(n):
    return np.random.rand(n)


# Funkce pro přímé řešení soustavy
def direct_solve(A, b):
    return np.linalg.solve(A, b)


# Funkce pro iterační řešení soustavy pomocí metody relaxace
def iterative_solve(A, b, w=1, tol=0.0001, max_iter=4):
    n = len(A)
    x = np.zeros(n)
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i][j] * x[j]
            x_new[i] = w / A[i][i] * s + (1 - w) * x[i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


# Měření času pro obě metody pro různé velikosti matice
matrix_sizes = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,300,400]
direct_times = []
iterative_times1 = []
iterative_times2 = []
iterative_times3 = []
iterative_times4 = []

for n in matrix_sizes:
    A = generate_matrix(n)
    b = generate_b(n)

    start = time.time()
    x = direct_solve(A, b)
    end = time.time()
    direct_times.append(end - start)

    start = time.time()
    x = iterative_solve(A, b,max_iter=5)
    end = time.time()
    iterative_times1.append(end - start)

    start = time.time()
    x = iterative_solve(A, b,max_iter=15)
    end = time.time()
    iterative_times2.append(end - start)

    start = time.time()
    x = iterative_solve(A, b,max_iter=50)
    end = time.time()
    iterative_times3.append(end - start)

    start = time.time()
    x = iterative_solve(A, b,max_iter=100)
    end = time.time()
    iterative_times4.append(end - start)

# Vykreslení grafu
plt.plot(matrix_sizes, direct_times, label="Direct")
plt.plot(matrix_sizes, iterative_times1, label="Iterative n = 5")
plt.plot(matrix_sizes, iterative_times2, label="Iterative n = 15")
plt.plot(matrix_sizes, iterative_times3, label="Iterative n = 50")
plt.plot(matrix_sizes, iterative_times4, label="Iterative n = 100")
plt.xlabel("Matrix size")
plt.ylabel("Time (s)")
plt.title("Comparison of direct and iterative linear system solvers")
plt.legend()
plt.show()