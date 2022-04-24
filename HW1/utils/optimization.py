import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
import timeit

warnings.filterwarnings("ignore")

def generate_particle(n_particle,n_space=95, n_select=30):
    return np.random.randint(n_space, size=(2,n_select,n_particle))

def index_to_solution(solution_space, idx):
    return solution_space[:,idx[0,:],idx[1,:]]

def index_to_cost(train_solution_space, train_y, idx):
    cur_x = index_to_solution(train_solution_space, idx)
    return calculate_cost(cur_x, train_y)

def calculate_cost(cur_x, train_y):
    clf = LogisticRegression().fit(cur_x,train_y)
    pred_y = clf.predict(cur_x)
    return sum(1*(pred_y == train_y)) / train_y.size

def cal_fitness(new_population, train_solution_space, train_y, num_sol):
    fitness = np.zeros((num_sol,1))
    for i_sol in range(num_sol):
        fitness[i_sol,:] = index_to_cost(train_solution_space, train_y, new_population[:,:,i_sol])
    return fitness

def Simulated_Annealing(train_solution_space, train_y, n_select=30):
    n_space = train_solution_space.shape[1]
    n_iter = 1000
    reduce_factor = 0.98

    # [Init] initialize random particle
    idx = generate_particle(1)[:,:,0]
    cur_cost = index_to_cost(train_solution_space, train_y, idx)

    # [Init] initialize best
    best_cost = cur_cost.copy()
    best_idx = idx.copy()
    temp = cur_cost.copy()

    start_time = timeit.default_timer()
    time_list = []

    n_choice = n_select//2
    for i_iter in range(n_iter):
        # [Perturb]
        next_idx = idx.copy()
        for i_choice in range(n_choice):
            if np.random.randint(2) == 1:
                next_idx[0,i_choice] = (next_idx[0,i_choice] + np.random.randint(n_space)) % n_space
            else:
                next_idx[1,i_choice] = (next_idx[1,i_choice] + np.random.randint(n_space)) % n_space
        next_cost = index_to_cost(train_solution_space, train_y, next_idx)
        
        # print best and save time info.
        print('SA Iter=%5d has cost=%f' % (i_iter, best_cost), end='\r')
        time_list.append([timeit.default_timer()-start_time,best_cost])

        # [Update]
        if next_cost > cur_cost: # if the cost is higher, update
            idx = next_idx
            cur_cost = next_cost
            # if the cost is higher then the best, replace the best idx and cost
            if next_cost > best_cost: 
                best_idx = next_idx.copy()
                best_cost = next_cost
        else: # the next cost is lower, update with probability
            p = np.exp((next_cost-cur_cost)/temp)
            p = 1 if p > 1 else p
            # print('-------------------------------------------------------------------- %f %f %f %f' % (cur_cost,next_cost,temp,p))
            if np.random.rand(1) < p: 
                idx = next_idx
                cur_cost = next_cost
        # reduce temp.
        temp = reduce_factor * temp
    print('')
    return best_idx, time_list


def Genetic_algorithm(train_solution_space, train_y, n_select=30):
    n_space = train_solution_space.shape[1]
    num_sol = 20
    num_generations = 50

    num_parents_mating = num_sol//2
    num_mutations = 2
        
    new_population = generate_particle(num_sol)
    idx = new_population[:,:,0]
    cur_cost = index_to_cost(train_solution_space, train_y, idx)
    best_solution = idx.copy()
    best_fitness = cur_cost.copy()

    start_time = timeit.default_timer()
    time_list = []
    for generation in range(num_generations):
        # [Fitness] Measuring the fitness of each chromosome in the population.
        fitness = cal_fitness(new_population, train_solution_space, train_y, num_sol)
        if best_fitness < np.max(fitness):
            best_fitness = np.max(fitness)
            best_idx = np.argmax(fitness)
            best_solution = new_population[:,:,best_idx]
            
        # print best and save time info.
        print("Generation %3d has Fitness=%.10f " % (generation, best_fitness),end='\r')
        time_list.append([timeit.default_timer()-start_time,best_fitness])

        # [Select] Selecting the best parents in the population for mating
        select_idx = np.argsort(fitness[:,0])[-1:-num_parents_mating-1:-1]
        parents = new_population[:,:,select_idx]
        
        # [Crossover] Generating next generation using crossover.
        offspring_size=(2,n_select,num_sol-num_parents_mating)
        offspring_crossover = np.empty(offspring_size,dtype='int')
        crossover_point = offspring_size[1]//2 # The point crossover takes place between two parents	
        for k in range(offspring_size[2]):
            # Index of the first & second parent
            parent1_idx = k % num_parents_mating
            parent2_idx = (k+1) % num_parents_mating
            # single point crossover by parents
            offspring_crossover[:, :crossover_point,k] = parents[:,:crossover_point,parent1_idx]
            offspring_crossover[:, crossover_point:,k] = parents[:,crossover_point:,parent2_idx]	

        # [Mutation] adding some variations to the offspring using mutation.
        mutations_counter = n_select // num_mutations
        for idx in range(offspring_size[2]):
            perturb_idx = np.random.randint(n_select)
            offspring_crossover[0,perturb_idx,idx] = (offspring_crossover[0,perturb_idx,idx] + np.random.randint(n_space)) % n_space
            offspring_crossover[1,perturb_idx,idx] = (offspring_crossover[1,perturb_idx,idx] + np.random.randint(n_space)) % n_space
        
        # [Update]
        new_population[:,:,:num_parents_mating] = parents
        new_population[:,:,num_parents_mating:] = offspring_crossover
    print('')
    return best_solution, time_list

def Particle_swarm(train_solution_space, train_y, n_select=30):
    n_space = train_solution_space.shape[1]
    n_particle = 20
    n_iter = 25
    n_mix = n_select//2
    particle = generate_particle(n_particle)
    best_cost = -np.inf
    best_particle = particle[:,:,0]
    start_time = timeit.default_timer()
    time_list = []
    cur_cost = cal_fitness(particle, train_solution_space, train_y, n_particle)
    for i_iter in range(n_iter):
        P_best = np.max(cur_cost)
        best_idx = np.argmax(cur_cost)
        cur_best_particle = particle[:,:,best_idx]
        if best_cost < P_best:
            best_cost = P_best
            best_particle = cur_best_particle.copy()
        print("Iteration %d has cost=%.10f " % (i_iter, best_cost),end='\r')
        time_list.append([timeit.default_timer()-start_time,best_cost])
        next_particle = np.zeros(particle.shape,dtype='int')
        for i_particle in range(n_particle):
            particle_type1 = np.zeros((2,n_select),dtype='int') # Random
            particle_type2 = np.zeros((2,n_select),dtype='int') # Mix GB
            particle_type3 = np.zeros((2,n_select),dtype='int') # Mix LB
            if np.random.randint(2) == 1:
                particle_type1[0,:] = (particle[0,:,i_particle] + np.random.randint(n_space)) % n_space
            else:
                particle_type1[1,:] = (particle[1,:,i_particle] + np.random.randint(n_space)) % n_space
            particle_type2[:,:n_mix] = best_particle[:,:n_mix]
            particle_type2[:,n_mix:] = particle[:,n_mix:,i_particle]
            particle_type3[:,:n_mix] = cur_best_particle[:,:n_mix]
            particle_type3[:,n_mix:] = particle[:,n_mix:,i_particle]
            fitness_type1 = index_to_cost(train_solution_space, train_y, particle_type1)
            fitness_type2 = index_to_cost(train_solution_space, train_y, particle_type2)
            fitness_type3 = index_to_cost(train_solution_space, train_y, particle_type3)
            if fitness_type2 > fitness_type1 and fitness_type2 > cur_cost[i_particle]:
                next_particle[:,:,i_particle] = particle_type2
            elif fitness_type3 > cur_cost[i_particle]:
                next_particle[:,:,i_particle] = particle_type3
            else:
                next_particle[:,:,i_particle] = particle_type1
            if fitness_type3 > cur_cost[i_particle]:
                next_particle[:,:,i_particle] = particle_type3
                cur_cost[i_particle] = fitness_type3
        particle = next_particle.copy()
    print('')
    return best_particle, time_list

def Ant_colony(train_solution_space, train_y, n_select=30):
    # seperate into N segments, find with n_ants in each iteration
    n_space = train_solution_space.shape[1]
    n_ant = 20
    n_iter = 50
    n_total = int(1e5)
    particle_pool = generate_particle(n_total)
    particle = generate_particle(n_ant)
    pheromone = np.ones(n_total)
    best_cost = -np.inf
    start_time = timeit.default_timer()
    time_list = []
    for i_iter in range(n_iter):
        ant_idx = np.random.choice(n_total, n_ant, p=pheromone/np.sum(pheromone)) # normalized pheromone
        ant_selection = particle_pool[:,:,ant_idx]

        ant_cost = np.zeros(n_ant)
        for i_ant in range(n_ant):
            ant_cost[i_ant] = index_to_cost(train_solution_space, train_y, ant_selection[:,:,i_ant])
        cur_best_cost = np.max(ant_cost)
        best_idx = np.argmax(ant_cost)
        if best_cost < cur_best_cost:
            best_cost = cur_best_cost
            best_ant_select = ant_selection[:,:,best_idx]
        print("Iteration %d has cost=%.10f " % (i_iter, best_cost),end='\r')
        time_list.append([timeit.default_timer()-start_time,best_cost])
        worst_cost = np.min(ant_cost)
        pheromone = pheromone / 2
        pheromone[best_idx] += cur_best_cost / worst_cost * 2
    print('')
    return best_ant_select, time_list

def Differential_evolution(train_solution_space, train_y, n_select=30):
    n_space = train_solution_space.shape[1]
    num_sol = 20
    num_generations = 50

    new_population = generate_particle(num_sol)
    idx = new_population[:,:,0]
    cur_cost = index_to_cost(train_solution_space, train_y, idx)
    best_solution = idx.copy()
    best_fitness = cur_cost.copy()
    start_time = timeit.default_timer()
    time_list = []
    fitness = cal_fitness(new_population, train_solution_space, train_y, num_sol)

    for generation in range(num_generations):
        # [Fitness] 
        if best_fitness < np.max(fitness):
            best_fitness = np.max(fitness)
            best_idx = np.argmax(fitness)
            best_solution = new_population[:,:,best_idx]
        print("Generation %3d has Fitness=%.10f " % (generation, best_fitness),end='\r')
        time_list.append([timeit.default_timer()-start_time,best_fitness])
        next_population = np.zeros(new_population.shape,dtype='int')

        for i_sol in range(num_sol):
            # [Select]
            idx_a = np.random.randint(0, num_sol)
            idx_b = np.random.randint(0, num_sol)
            idx_c = np.random.randint(0, num_sol)
            # [Crossover]
            if np.random.randint(2) == 1:
                next_population[0,:,i_sol] = (new_population[0,:,idx_a] + (new_population[0,:,idx_b]-new_population[0,:,idx_c])) % n_space
            else:
                next_population[1,:,i_sol] = (new_population[1,:,idx_a] + (new_population[1,:,idx_b]-new_population[1,:,idx_c])) % n_space
        # [Update]
        next_fitness = cal_fitness(next_population, train_solution_space, train_y, num_sol)
        for i_sol in range(num_sol):
            if next_fitness[i_sol] > fitness[i_sol]:
                new_population[:,:,i_sol] = next_population[:,:,i_sol]
                fitness[i_sol] = next_fitness[i_sol]
    print('')
    return best_solution, time_list