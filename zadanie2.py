import genetic_all as ga  
import matplotlib.pyplot as plt
import numpy as np

# Установка параметров
NUM_GEN = 1000  # Количество поколений
NUM_RUNS = 5  # Количество запусков

# Описание пространства генов
space = ga.uniform_space(amount_of_genes=10, lower_limit=-800, upper_limit=800)
amp = space[1,:] / 100  # Амплитуда для мутации

# Создание графика
plt.figure(figsize=(10, 6))
eliteSize=5
popSize=100
ostatne = popSize - eliteSize
# Многократный запуск GA
for run in range(NUM_RUNS):
    # Инициализация популяции
    pop = ga.genrpop(pop_size=100, space=space)  # Увеличиваем размер популяции
    print(f"Run {run+1} - Initial population:\n{pop}\n")
    
    evolution = []  # Массив для отслеживания минимальной fitness функции

    # Эволюция
    for i in range(NUM_GEN):
        fitness = ga.testfn3b(pop)
        evolution.append(fitness.min())  # Добавление минимальной fitness функции на текущем поколении
        best_pop, _ = ga.selbest(pop, fitness, n_list=[eliteSize], reverse = False)  # Отбор 3 лучших особей (повышаем разнообразие)
        muta_pop2, _ = ga.seltourn(pop, fitness, n=ostatne//2, reverse = False)
        muta_pop, _ = ga.selrand(pop, fitness, n=ostatne//2) 
        
        muta_pop3 = np.vstack((muta_pop,muta_pop2)) 
    
        krizenie_pop = ga.crossov(muta_pop3, 2, mode=0) 
        muta_pop1 = ga.mutx(krizenie_pop, rate=0.06,space = space)  
        
        pop = np.vstack((best_pop,  muta_pop1))  
    print(f"Run {run+1} - Best solution: {best_pop[0]}")
    print(f"Run {run+1} - Best fitness value: {fitness.min()}")

    # Построение графика для текущего запуска
    plt.plot(evolution, label=f'Run {run+1}')

# Настройка графика
plt.xlabel('Generations')
plt.ylabel('Fitness Value')
plt.title('GA Fitness Evolution Across Multiple Runs')
plt.legend()
plt.grid(True)
plt.show()
