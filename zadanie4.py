import genetic_all as ga
import numpy as np
import matplotlib.pyplot as plt

space = np.array([[0]*5, [3000000]*5])
populacia = 350
pocetElit = 14
krizenie = populacia - pocetElit
iteracia = 1000

def death_penalty(pop):
    fitness_values_death = []
    for x in pop:
        J = 0.04*x[0] + 0.07*x[1] + 0.11*x[2] + 0.06*x[3] + 0.05*x[4] #ucelova funkcia max
        if (sum(x) > 10000000 or x[0] + x[1] > 2500000 or x[4] > x[3] or 
            (-0.5)*x[0] - 0.5*x[1] + 0.5*x[2] + 0.5*x[3] - 0.5*x[4] > 0):
            fitness_values_death.append(-float("inf"))
        else:
            fitness_values_death.append(J)
    return fitness_values_death

def step_penalty(pop):
    fitness_values_step = []
    hodnota = 10000001
    for x in pop:
        pocet_penalty = 0
        J = 0.04*x[0] + 0.07*x[1] + 0.11*x[2] + 0.06*x[3] + 0.05*x[4]
        if np.sum(x) > 10000000:
            pocet_penalty += 1
        if x[4] > x[3]:
            pocet_penalty += 1
        if (-0.5)*x[0] - 0.5*x[1] + 0.5*x[2] + 0.5*x[3] - 0.5*x[4] > 0:
            pocet_penalty += 1
        if x[0] + x[1] > 2500000:
            pocet_penalty += 1
        penalty = pocet_penalty * hodnota
        fitness_values_step.append(J - penalty)
    return fitness_values_step

def umerne_penalty(pop):
    fitness_values_umerne = []
    a = 0
    b = 1
    c = 1
    for x in pop:
        J = 0.04*x[0] + 0.07*x[1] + 0.11*x[2] + 0.06*x[3] + 0.05*x[4]
        p1 = x[0] + x[1] + x[2] + x[3] + x[4] - 10000000  
        p2 = x[0] + x[1] - 2500000  
        p3 = -x[3] + x[4]  
        p4 = (-0.5)*x[0] - 0.5*x[1] + 0.5*x[2] + 0.5*x[3] - 0.5*x[4] 
        if p1 > 0:
            J -= (a + c*(p1**b))
        if p2 > 0:
            J -= (a + c*(p2**b))
        if p3 > 0:
            J -= (a + c*(p3**b))
        if p4 > 0:
            J -= (a + c*(p4**b))
        fitness_values_umerne.append(J)
    return fitness_values_umerne

def GAlgorithm(penalty_type='step'):#aj tu sme mozeme zmenit pokutu(114)
    pts = 2
    mode = 0
    rate = 0.06
    global bestVector, bestValue

    pop = np.random.randint(0, 2500000, size=(populacia, 5))
    ilustration = []
    best_value = -1e9

    if penalty_type == 'step':
        fitness_func = step_penalty
    elif penalty_type == 'umerne':
        fitness_func = umerne_penalty
    elif penalty_type == 'death':
        fitness_func = death_penalty
   

    for i in range(iteracia):
        fitnessedPop = fitness_func(pop)

        current_best_idx = np.argmax(fitnessedPop)
        current_best_value = fitnessedPop[current_best_idx]

        if current_best_value > best_value:
            best_value = current_best_value
            bestVector = pop[current_best_idx].copy()

        ilustration.append([i, best_value])

        elite_pop, _ = ga.selbest(pop, fitnessedPop, [pocetElit], reverse=True)
        remaining_pop, _ = ga.seltourn(pop, fitnessedPop, krizenie, reverse=True)

        new_pop = ga.crossov(remaining_pop, pts, mode)
        new_pop = ga.mutx(new_pop, rate, space)

        pop = np.vstack((elite_pop, new_pop))

    ilustration = np.array(ilustration)

    if ilustration.shape[0] > 0:
        
        plt.xlabel("Iteration")
        plt.ylabel("Function Value")
        plt.plot(ilustration[:, 0], ilustration[:, 1])
        plt.grid()

    return best_value


bestValue = float('-inf')
for i in range(5):
    currentValue = GAlgorithm(penalty_type='step')#tu sme mozeme menit pokutu(62)
    if currentValue > bestValue:
        bestValue = currentValue
    print("Best solution:", bestVector)
    print(f"Run {i+1}: Best fitness value: {currentValue}")
plt.legend()
plt.show()
