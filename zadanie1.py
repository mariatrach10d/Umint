import numpy as np
import matplotlib.pyplot as plt
import random


def schwefel(x):
    return -x * np.sin(np.sqrt(abs(x)))

step_size = 10  
max_iterations = 1000  # Maximálny počet iterácií
num_restarts = 20  # Počet opakovaní hľadania

random_local_min = None  # Prvý náhodne nájdený lokálny minim

# Generovanie grafu
x = np.linspace(-800, 800, 1000)
y = schwefel(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label="Schwefel Function", color="blue")

# Spustenie viacerých hľadaní minima
for _ in range(num_restarts):
    current_x = random.uniform(-800, 800)  # Náhodný začiatočný bod
    current_y = schwefel(current_x)
    
    step = step_size

    for _ in range(max_iterations):
        left_x = max(-800, current_x - step)  # Kontrola ľavej hranice
        right_x = min(800, current_x + step)  # Kontrola pravej hranice

        left_y = schwefel(left_x)
        right_y = schwefel(right_x)

        # Výber najlepšieho kroku
        if left_y < current_y:
            current_x, current_y = left_x, left_y
        elif right_y < current_y:
            current_x, current_y = right_x, right_y
        else:
            # Uloženie prvého nájdeného lokálneho minima a ukončenie
            if random_local_min is None:
                random_local_min = (current_x, current_y)
            break  # Ukončenie cyklu, keď sa nájde lokálne minimum

    if random_local_min:
        break  # Ukončenie hľadania po nájdení prvého lokálneho minima

# Zobrazenie prvého náhodne nájdeného lokálneho minima
if random_local_min:
    plt.scatter([random_local_min[0]], [random_local_min[1]], c="orange", marker="s", s=150, label="First Random Local Min")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
