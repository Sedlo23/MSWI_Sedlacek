import timeit
import numpy as np
import scipy.integrate as integrate
import networkx as nx
import math
from random import randint
import matplotlib.pyplot as plt

# 1. Skalární součin
def dot_product_python(a, b):
    return sum(i*j for i, j in zip(a, b))

def dot_product_numpy(a, b):
    return np.dot(a, b)

a = [randint(0, 100) for i in range(1000)]
b = [randint(0, 100) for i in range(1000)]

dot_product_python_time = timeit.timeit('dot_product_python(a, b)', globals=globals(), number=1000)
dot_product_numpy_time = timeit.timeit('dot_product_numpy(a, b)', globals=globals(), number=1000)

# 2. Určitý integrál
def integral_python(f, a, b):
    n = 1000
    dx = (b - a) / n
    return sum(f(a + i*dx) for i in range(n)) * dx

def integral_scipy(f, a, b):
    return integrate.quad(f, a, b)[0]

def f(x):
    return x**2

integral_python_time = timeit.timeit('integral_python(f, 0, 1)', globals=globals(), number=1000)
integral_scipy_time = timeit.timeit('integral_scipy(f, 0, 1)', globals=globals(), number=1000)

# 3. Nejkratší cesta v grafu
def shortest_path_python(graph, start, end):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == end:
                return path + [next]
            else:
                queue.append((next, path + [next]))

def shortest_path_networkx(graph, start, end):
    return nx.shortest_path(graph, start, end)

graph = {0: {1, 2}, 1: {2, 3}, 2: {3}, 3: {4}, 4: {0}}
G = nx.Graph(graph)

shortest_path_python_time = timeit.timeit('shortest_path_python(graph,0, 4)', globals=globals(), number=1000)
shortest_path_networkx_time = timeit.timeit('shortest_path_networkx(G, 0, 4)', globals=globals(), number=1000)

# 4. Determinant matice
def determinant_python(matrix):
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    determinant = 0
    for c in range(len(matrix)):
        submatrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        determinant += ((-1) ** c) * matrix[0][c] * determinant_python(submatrix)
        return determinant

def determinant_numpy(matrix):
    return np.linalg.det(matrix)

matrix = [[randint(0, 10) for i in range(4)] for j in range(4)]
matrix_np = np.array(matrix)

determinant_python_time = timeit.timeit('determinant_python(matrix)', globals=globals(), number=1000)
determinant_numpy_time = timeit.timeit('determinant_numpy(matrix_np)', globals=globals(), number=1000)

#5. Řešení soustavy lineárních rovnic
def solve_linear_system_python(A, b):
    return [round(x, 2) for x in np.linalg.solve(A, b)]

def solve_linear_system_numpy(A, b):
    return np.linalg.solve(A, b)

A = [[randint(1, 10) for i in range(3)] for j in range(3)]
b = [randint(1, 10) for i in range(3)]
A_np = np.array(A)
b_np = np.array(b)

solve_linear_system_python_time = timeit.timeit('solve_linear_system_python(A, b)', globals=globals(), number=1000)
solve_linear_system_numpy_time = timeit.timeit('solve_linear_system_numpy(A_np, b_np)', globals=globals(), number=1000)

tasks = ['Skalární součin', 'Určitý integrál', 'Nejkratší cesta', 'Determinant matice', 'Řešení soustavy']
python_times = [dot_product_python_time, integral_python_time, shortest_path_python_time, determinant_python_time, solve_linear_system_python_time]
library_times = [dot_product_numpy_time, integral_scipy_time, shortest_path_networkx_time, determinant_numpy_time, solve_linear_system_numpy_time]

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, python_times, width, label='Python')
rects2 = ax.bar(x + width/2, library_times, width, label='Knihovny')

ax.set_ylabel('Čas (s)')
ax.set_title('Porovnání rychlosti Pythonu a specializovaných knihoven')
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 4)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()