import timeit
import numpy as np


# Definice dvou vektorů
for n in range(1, 12):
    print("n=",n)
    # Generování náhodných vektorů
    a = np.random.rand(n)
    b = np.random.rand(n)

    # Výpočet skalárního součinu pomocí standardního Pythonu
    def scalar_product(a, b):
        return sum([a[i] * b[i] for i in range(len(a))])

    # Výpočet skalárního součinu pomocí knihovny numpy
    def numpy_scalar_product(a, b):
        return np.dot(a, b)

    # Měření času pro standardní Python
    standard_time = timeit.timeit(lambda: scalar_product(a, b), number=1000000)
    print("Standard time:", standard_time)


    # Měření času pro numpy
    numpy_time = timeit.timeit(lambda: numpy_scalar_product(a, b), number=1000000)
    print("Numpy time:", numpy_time)