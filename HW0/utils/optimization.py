import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.random.seed(10)

def predict(train_x,coeff):
    return 1*(train_x@coeff > 0.5)

def calculate_cost(coeff, train_x, train_y, type='discrete'):
    # print(coeff.shape, train_x.shape, train_y.shape)
    if type == 'continuous':
        return coeff.T@train_x.T@train_x@coeff - 2*coeff.T@train_x.T@train_y
    else:
        cur_pred = predict(train_x,coeff)
        return np.sum(train_y == cur_pred) / train_y.size

def Simulated_Annealing(train_x, train_y):
    # print('Start SA')
    # print('train_x & train_y has size', train_x.shape, train_y.shape)
    n_iter = int(1e3)
    reduce_factor = 0.5

    def perturb(coeff,step_size=1):
        return coeff + ((np.random.rand(coeff.shape[0],coeff.shape[1])*2-1)*step_size)

    coeff = np.ones((train_x.shape[1],1))
    best_coeff = coeff
    best_cost = calculate_cost(best_coeff, train_x, train_y) 
    no_improve_cnt = 0
    temp = best_cost
    step_size = 1
    for i_iter in range(n_iter):
        coeff_next = perturb(coeff,step_size)
        cur_cost = calculate_cost(coeff, train_x, train_y) 
        print('SA Iter=%2d with Temp=%5.5f has cost=%f' % (i_iter, temp, best_cost), end='\r')
        next_cost = calculate_cost(coeff_next, train_x, train_y) 
        # print(cur_cost, next_cost, best_cost)
        if next_cost > cur_cost: # if the cost is higher, update coeff
            coeff = coeff_next
            if next_cost > best_cost: # if the cost is higher then the best, replace the best coeff and cost
                no_improve_cnt = 0
                best_coeff = coeff_next
                best_cost = next_cost
        else: # the next cost is lower, update with probability
            p = np.exp((cur_cost-next_cost)/temp)
            p = 1 if p > 1 else p
            if np.random.rand(1) < p: 
                coeff = coeff_next
        temp = reduce_factor * temp
        no_improve_cnt += 1
        if no_improve_cnt > 1e3:
            step_size /= 2
            no_improve_cnt = 0
    print('')
    return best_coeff


def Genetic_algorithm(train_x, train_y):
    def cal_fitness(new_population, train_x, train_y, num_sol):
        fitness = np.zeros((num_sol,1))
        for i_sol in range(num_sol):
            fitness[i_sol,:] = calculate_cost(new_population[i_sol:i_sol+1,:].T, train_x, train_y) 
        return fitness
    num_sol = 8
    num_parents_mating = 4
    num_mutations=2
        
    num_weights = train_x.shape[1]
    pop_size = (num_sol,num_weights) 
    
    num_generations = 100
    new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
    best_solution = new_population[0:1,:].T
    best_fitness = calculate_cost(best_solution, train_x, train_y)
    for generation in range(num_generations):
        # [Fitness] Measuring the fitness of each chromosome in the population.
        fitness = cal_fitness(new_population, train_x, train_y, num_sol)
        if best_fitness < np.max(fitness):
            best_fitness = np.max(fitness)
            best_idx = np.argmax(fitness)
            best_solution = new_population[best_idx:best_idx+1,:].copy().T
        print("Generation %d has Fitness=%.10f " % (generation, best_fitness),end='\r')

        # [Select] Selecting the best parents in the population for mating
        select_idx = np.argsort(fitness[:,0])[-1:-num_parents_mating-1:-1]
        parents = new_population[select_idx,:]
        
        # [Crossover] Generating next generation using crossover.
        offspring_size=(num_sol-num_parents_mating, num_weights)
        offspring_crossover = np.empty(offspring_size)
        crossover_point = offspring_size[1]//2 # The point crossover takes place between two parents	
        for k in range(offspring_size[0]):
            # Index of the first & second parent
            parent1_idx = k % num_parents_mating
            parent2_idx = (k+1) % num_parents_mating
            # single point crossover by parents
            offspring_crossover[k, :crossover_point] = parents[parent1_idx, :crossover_point]
            offspring_crossover[k, crossover_point:] = parents[parent2_idx, crossover_point:]	
           
        # [Mutation] adding some variations to the offspring using mutation.
        mutations_counter = num_weights // num_mutations
        for idx in range(offspring_size[0]):
            gene_idx = mutations_counter - 1
            for mutation_num in range(num_mutations):
                # The random value to be added to the gene.
                random_value = np.random.uniform(-1.0, 1.0, 1)
                offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
                gene_idx = gene_idx + mutations_counter	
        
        # [Update]
        new_population[:num_parents_mating, :] = parents
        new_population[num_parents_mating:, :] = offspring_crossover
    print('')
    return best_solution

def Particle_swarm(train_x, train_y):
    n_particle = 100
    n_iter = 100
    particle = np.random.uniform(low=-4.0, high=4.0, size=(train_x.shape[1],n_particle)) # coeff_size x n_particle
    velocity = np.zeros((train_x.shape[1],n_particle))
    best_cost = -np.inf
    for i_iter in range(n_iter):
        cur_cost = np.zeros(n_particle)
        for i_particle in range(n_particle):
            cur_cost[i_particle] = calculate_cost(particle[:,i_particle:i_particle+1], train_x, train_y)
            # check the size
        P_best = np.max(cur_cost)
        best_idx = np.argmax(cur_cost)
        cur_best_particle = particle[:,best_idx:best_idx+1]
        if best_cost < P_best:
            best_cost = P_best
            best_particle = cur_best_particle.copy()
        P_worst = np.min(cur_cost)
        print("Iteration %d has cost=%.10f " % (i_iter, best_cost),end='\r')
        r2 = np.random.uniform(low=0.0, high=1.0)
        velocity = velocity + r2*(cur_best_particle-particle)
        particle = particle + velocity
    print('')
    return best_particle

def Ant_colony(train_x, train_y):
    # seperate into N segments, find with n_ants in each iteration
    n_iter = 100
    n_ant = 100
    n_select = int(1e5)
    total_selection = np.random.uniform(low=-4.0, high=4.0, size=(train_x.shape[1],n_select)) # coeff_size x n_particle
    pheromone = np.ones(n_select)
    best_cost = -np.inf
    for i_iter in range(n_iter):
        ant_idx = np.random.choice(n_select, n_ant, p=pheromone/np.sum(pheromone)) # normalized pheromone
        ant_selection = total_selection[:,ant_idx]
        ant_cost = np.zeros(n_ant)
        for i_ant in range(n_ant):
            ant_cost[i_ant] = calculate_cost(ant_selection[:,i_ant:i_ant+1], train_x, train_y)
        cur_best_cost = np.max(ant_cost)
        best_idx = np.argmax(ant_cost)
        if best_cost < cur_best_cost:
            best_cost = cur_best_cost
            best_ant_select = ant_selection[:,best_idx:best_idx+1].copy()
        print("Iteration %d has cost=%.10f " % (i_iter, best_cost),end='\r')
        worst_cost = np.min(ant_cost)
        pheromone = pheromone / 2
        pheromone[best_idx] += cur_best_cost / worst_cost * 2
    print('')
    return best_ant_select

def Differential_evolution(train_x, train_y):
    def cal_fitness(new_population, train_x, train_y, num_sol):
        fitness = np.zeros((num_sol,1))
        for i_sol in range(num_sol):
            fitness[i_sol,:] = calculate_cost(new_population[i_sol:i_sol+1,:].T, train_x, train_y) 
        return fitness
    num_sol = 100
    r = 0.9
        
    num_weights = train_x.shape[1]
    pop_size = (num_sol,num_weights) 
    
    num_generations = 100
    new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
    best_solution = new_population[0:1,:].T
    best_fitness = calculate_cost(best_solution, train_x, train_y)
    for generation in range(num_generations):
        # [Fitness] 
        fitness = cal_fitness(new_population, train_x, train_y, num_sol)
        if best_fitness < np.max(fitness):
            best_fitness = np.max(fitness)
            best_idx = np.argmax(fitness)
            best_solution = new_population[best_idx:best_idx+1,:].copy().T
        print("Generation %d has Fitness=%.10f " % (generation, best_fitness),end='\r')
        next_population = np.zeros(new_population.shape)
        for i_sol in range(num_sol):
            # [Select]
            idx_a = np.random.randint(0, num_sol)
            idx_b = np.random.randint(0, num_sol)
            idx_c = np.random.randint(0, num_sol)
            # [Crossover]
            next_population[i_sol,:] = new_population[idx_a,:] + r*(new_population[idx_b,:]-new_population[idx_c,:])
        # [Update]
        next_fitness = cal_fitness(next_population, train_x, train_y, num_sol)
        for i_sol in range(num_sol):
            if next_fitness[i_sol] > fitness[i_sol]:
                new_population[i_sol,:] = next_population[i_sol,:]
    print('')
    return best_solution