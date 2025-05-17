import genetic_all as ga  
import matplotlib.pyplot as plt
import numpy as np

NUM_GEN = 1000  
NUM_RUNS = 5  
# Generuje space s rovnakymi obmedzeniami pre vsetky geny
space = ga.uniform_space(amount_of_genes=10, lower_limit=-800, upper_limit=800)
amp = space[1,:] / 100 
plt.figure(figsize=(10, 6))
eliteSize=5
popSize=100
ostatne = popSize - eliteSize
for run in range(NUM_RUNS):
    # generovanie novej populacie
    pop = ga.genrpop(pop_size=100, space=space) 
    
    evolution = []  

    for i in range(NUM_GEN):
        fitness = ga.testfn3b(pop)
        evolution.append(fitness.min())  # Добавление минимальной fitness функции на текущем поколении
        # elitaristicky vyber
        # funkcia zoradi jedince podla uspe
        best_pop, _ = ga.selbest(pop, fitness, n_list=[eliteSize], reverse = False) 
        # turnajovy vyber
        # funkcia vykona "zapasy" medzi nahodne vybranymi jedincami
        # lepsi jedinec bude skopirovany do novej pop
        muta_pop1, _ = ga.seltourn(pop, fitness, n=ostatne//2, reverse = False)
        # nahodny vyber
        # funkcia vytvori novu pop z nahodnych jedincov
        muta_pop2, _ = ga.selrand(pop, fitness, n=ostatne//2) 
        # spája polia
        muta_pop = np.vstack((muta_pop1,muta_pop2)) 
        # krizenie s vyberom poctu bodov krizenia
        # funkcia zkrizi pary jedincov tak, ze sa podla parametra pts
        # vytvoria body krizenia medzi ktorymi sa vymenia geny jedincov
        krizenie_pop = ga.crossov(muta_pop, 2, mode=0) 
        # obycajna mutacia
        # funckia zmeni nahodne vybrane geny na cisla
        # v rozsahu danom parametrom space
        # intenzita mutacie je v parametri rate
        muta_pop1 = ga.mutx(krizenie_pop, rate=0.06,space = space)  
        
        pop = np.vstack((best_pop,  muta_pop1))  
    print(f"{run+1} - Best solution: {best_pop[0]}")
    print(f"{run+1} - Best fitness value: {fitness.min()}")

    plt.plot(evolution, label=f' {run+1}')

plt.xlabel('Generations')
plt.ylabel('Fitness Value')
plt.legend()
plt.grid(True)
plt.show()
