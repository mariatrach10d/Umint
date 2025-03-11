import numpy as np
import matplotlib.pyplot as plt
import random

# definujweme funkciu 
def schwefel(x):
    return -x * np.sin(np.sqrt(abs(x)))


step_size = 10 
max_iterations = 1000  
num_restarts = 20 

best_x = None
best_y = float("inf")  # nekonecnost pre porovnanie

local_minima = []  # zoznam lok min

# generuje grafik
x = np.linspace(-800, 800, 1000)
y = schwefel(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y, label="Schwefel Function", color="blue")

# spustame viackratne hladanie min
for _ in range(num_restarts):
    current_x = random.uniform(-800, 800)  # randomny zaciatocny bod
    current_y = schwefel(current_x)
    
    x_history = [current_x]
    y_history = [current_y]
    
    step = step_size

    for _ in range(max_iterations):
        left_x = max(-800, current_x - step)#oznacovania hranici
        right_x = min(800, current_x + step)

        left_y = schwefel(left_x)
        right_y = schwefel(right_x)

        # najlepsi step
        if left_y < current_y:
            current_x, current_y = left_x, left_y
        elif right_y < current_y:
            current_x, current_y = right_x, right_y
        else:
            found_similar_minimum = False  # Флаг, который показывает, найден ли похожий минимум

            for x_min, _ in local_minima:  # pozriem na vsetky lok minimu
                if abs(current_x - x_min) < 5:  # overujeme nakolko terazsi minimum blizko k inemu
                    found_similar_minimum = True  
                    break  

            if not found_similar_minimum:  
                local_minima.append((current_x, current_y))  
 
        
        x_history.append(current_x)
        y_history.append(current_y)

    # overovanie glob min
    if current_y < best_y:
        best_x, best_y = current_x, current_y

    plt.scatter(x_history, y_history, s=30)  # ukazeme vsetky body ktori sme pouzivaali

# ukazuje lok min
for x_min, y_min in local_minima:
    plt.scatter([x_min], [y_min], c="green", marker="o", s=80, label="Local Min" )
# ukazuje glob min
plt.scatter([best_x], [best_y], c="red", marker="*", s=200, label="Global Min")

plt.xlabel("x")
plt.ylabel("y")
plt.show()





