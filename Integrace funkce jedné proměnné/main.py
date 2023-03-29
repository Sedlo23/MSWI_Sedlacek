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


""""
Tento kód implementuje Eulerovu metodu pro Lorenzův systém a vizualizuje výsledky v 3D grafu.

Funkce euler_step(f, x, dt) představuje jeden krok Eulerovy metody, kde f je funkce reprezentující Lorenzův 
systém, x je aktuální stav systému a dt je časový krok. Funkce vrací nový stav systému po uplynutí časového kroku dt.

Funkce lorenz_system(x) reprezentuje Lorenzův systém. Parametry sigma, rho a beta jsou konstanty,
 které určují charakteristiky systému. Funkce vrací vektor obsahující rychlosti změn pro každou proměnnou systému.

V kódu se definuje časový krok dt a celková doba simulace T. Dále se vytváří pole t, které obsahuje
 časové body simulace.

Pole x obsahuje výsledky simulace. Na začátku jsou všechny hodnoty nulové, s výjimkou prvního časového bodu,
 kde jsou hodnoty proměnných nastaveny na [0.0, 1.0, 0.0], což jsou počáteční hodnoty systému.

V cyklu se pak počítá nový stav systému pro každý časový bod. Pro výpočet nového stavu se používá
Eulerova metoda a funkce lorenz_system.

Výsledné hodnoty jsou uloženy do pole x.

Na závěr se výsledky vizualizují v 3D grafu pomocí knihovny matplotlib.

Graf zobrazuje trajektorii Lorenzova systému v prostoru X, Y a Z. Lze pozorovat charakteristický tvar atraktoru - 
tzv. motýlí křídla. Lorenzův systém je chaotický, což znamená, že i malé změny v počátečních podmínkách mohou 
vést k výrazně odlišným výsledkům. To se odráží i na grafu, kde se trajektorie rychle rozptyluje."""