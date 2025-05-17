import genetic_all as ga
import numpy as np
import matplotlib.pyplot as plt
import random

B = np.array([[25, 68], [12, 75], [32, 17], [51, 64], [20, 19],
              [52, 87], [80, 37], [35, 82], [2, 15], [50, 90], [13, 50],
              [85, 52], [97, 27], [37, 67], [20, 82], [49, 0], [62, 14],
              [7, 60]])
num_points = len(B)

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def fitness(population):
    fitness_values = []
    
    for route in population:
        total_dist = 0
        first_point = B[route[0]]
        total_dist += euclidean_distance(0, 0, first_point[0], first_point[1])

        for i in range(len(route) - 1):
            p1, p2 = B[route[i]], B[route[i+1]]
            total_dist += euclidean_distance(p1[0], p1[1], p2[0], p2[1])

        last_point = B[route[-1]]
        total_dist += euclidean_distance(last_point[0], last_point[1], 100, 100)

        fitness_values.append(total_dist)
    
    return fitness_values

def initialize_population(size):
    return [[0] + random.sample(range(1, num_points - 1), num_points - 2) + [num_points - 1] for _ in range(size)]

def crossord(parent1, parent2):
    size = len(parent1)
    crossord_point1 = random.randint(0, size - 1)
    crossord_point2 = random.randint(crossord_point1, size - 1)
    
    child = [-1] * size
    
    for i in range(crossord_point1, crossord_point2 + 1):
        child[i] = parent1[i]
    
    current_index = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[current_index] in child:
                current_index += 1
            child[i] = parent2[current_index]
            current_index += 1
    
    return child

def genetic_algorithm(generations=100, pop_size=100, mutation_rate=0.03):
    pop = initialize_population(pop_size)
    best_route, best_fitness = None, float('inf')
    history = []
    
    for gen in range(generations):
        fitness_values = fitness(pop)
        best_fitness_gen = min(fitness_values)
        
        if best_fitness_gen < best_fitness:
            best_fitness = best_fitness_gen
            best_route = pop[fitness_values.index(best_fitness_gen)]
            history.append(best_fitness)
        
        elitePop, _ = ga.selbest(pop, fitness_values, [50], reverse=False)
        
        pop_array = np.array(pop)
        fitness_values_array = np.array(fitness_values)
        
        indices = list(range(len(pop)))
        ostatne_indices, _ = ga.selrand(np.array(indices), fitness_values_array, pop_size - 50)
        ostatne = [pop[i] for i in ostatne_indices]
        
        next_gen = np.vstack([elitePop,[crossord(route, random.choice(ostatne)) for route in ostatne]])
        ga.swapgen(next_gen, mutation_rate)
        pop = next_gen
        
        print(f"Generacia {gen+1}: Najlepsia cesta - {best_fitness:.2f}")
        
    return best_route, best_fitness, history

best_route, best_fitness, history = genetic_algorithm()
print(f"celkova dlzka cesty: {best_fitness:.2f}")


def plot_route(route):
    plt.figure(figsize=(10, 6))
    route_coords = np.vstack(([0, 0], B[route], [100, 100]))
    
    plt.scatter(B[:, 0], B[:, 1], color='blue', s=100, edgecolors='black')
    plt.plot(route_coords[:, 0], route_coords[:, 1], 'g-o')
    
    for i, (x, y) in enumerate(B, start=2):
        plt.text(x, y, str(i), fontsize=12, ha='right', color='black', weight='bold')
    
    
    plt.grid(True)
    plt.show()

plot_route(best_route)

plt.figure()
plt.plot(history)
plt.xlabel("generacia")
plt.ylabel("najlepsia cesta")
plt.title("evolucia")
plt.show()
