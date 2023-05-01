import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from threading import Timer


def generate_matrix(n):
    return np.random.uniform(-1, 1, (n, n))


def generate_b(n):
    return np.random.uniform(-1, 1, n)


def direct_solve(A, b):
    return np.linalg.solve(A, b)


def iterative_solve(A, b, w=1.5, tol=1e-5, max_time=120):
    n = len(A)
    x = np.zeros(n)
    start_time = time.time()
    k = 0
    while True:
        x_new = np.zeros(n)
        for i in range(n):
            s = (b[i])
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x[j]
            if A[i, i] != 0:
                x_new[i] = w / A[i, i] * s + (1 - w) * x[i]
            else:
                x_new[i] = x[i]
            if np.isnan(x_new[i]) or np.isinf(x_new[i]):
                return None, k  # Přerušit iteraci v případě neplatných hodnot nebo přetečení
        k += 1
        if np.linalg.norm(x_new - x) < tol or (time.time() - start_time) > max_time:
            break
        x = x_new
    return x, k



def run_with_timeout(func, timeout, *args, **kwargs):
    result = None
    timer = Timer(timeout, lambda: result)
    timer.start()
    try:
        result = func(*args, **kwargs)
    finally:
        timer.cancel()
    return result


def estimate_time(matrix_size, previous_matrix_size, previous_time, time_complexity_function):
    factor = time_complexity_function(matrix_size) / time_complexity_function(previous_matrix_size)
    return previous_time * factor


def time_complexity_iterative(n):
    return n ** 2  # Pro SOR iterační metodu, můžete upravit podle časové složitosti konkrétní iterační metody


matrix_sizes = [int(10 ** i) for i in range(1, 5)]  # Velikosti matice: 1E1, 1E2, 1E3, 1E4, 1E5, 1E6
direct_times = []
iterative_times = []
previous_iterative_time = None

for n in matrix_sizes:
    print(f"Processing matrix size: {n}")
    A = generate_matrix(n)
    b = generate_b(n)

    start = time.time()
    x_direct = direct_solve(A, b)
    end = time.time()
    direct_times.append(end - start)

    start = time.time()
    x_iter, iterations = iterative_solve(A, b, w=1, tol=0.1, max_time=120)
    end = time.time()

    if (end - start) >= 120:
        if previous_iterative_time is not None:
            estimated_time = estimate_time(n, previous_matrix_size, previous_iterative_time, time_complexity_iterative)
            iterative_times.append(estimated_time)
            print(f"Matrix size {n} took too long for iterative solve. Estimated time: {estimated_time:.2f}s")
        else:
            print(f"Matrix size {n} took too long for iterative solve. No previous data for estimation.")
            iterative_times.append(None)
    else:
        iterative_times.append(end - start)
        previous_iterative_time = end - start
        previous_matrix_size = n

plt.plot(matrix_sizes, direct_times, label="Direct")
plt.plot(matrix_sizes, iterative_times, label="Iterative (SOR)", linestyle='--')
plt.xlabel("Matrix size")
plt.ylabel("Average Time (s)")
plt.title("Comparison of direct and iterative linear system solvers")
plt.legend()
plt.show()