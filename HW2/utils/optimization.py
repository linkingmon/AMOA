from random import random
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from utils.preprocess import inorder_walk
import timeit

warnings.filterwarnings("ignore")

def generate_particle(n_particle,n_space=4560, n_select=10):
    return np.random.randint(n_space, size=(n_particle,n_select))

def index_to_solution(solution_space, idx):
    return solution_space[:,idx[:]]

def index_to_cost(train_solution_space, train_y, idx):
    cur_x = index_to_solution(train_solution_space, idx)
    return calculate_cost(cur_x, train_y)

def calculate_cost(cur_x, train_y):
    # print(cur_x.shape,train_y.shape)
    clf = LogisticRegression().fit(cur_x,train_y)
    pred_y = clf.predict(cur_x)
    return sum(1*(pred_y == train_y)) / train_y.size

def cal_fitness(new_population, train_solution_space, train_y, num_sol):
    fitness = np.zeros((num_sol,1))
    for i_sol in range(num_sol):
        fitness[i_sol,:] = index_to_cost(train_solution_space, train_y, new_population[i_sol,:])
    return fitness
    
def cal_fitness2(new_population, train_solution_space, train_y, num_sol):
    num_sample = train_y.shape[0]
    fitness = np.zeros((num_sol,num_sample))
    for i_sol in range(num_sol):
        fitness[i_sol,:] = index_to_cost2(train_solution_space, train_y, new_population[i_sol,:])
    return fitness

def index_to_cost2(train_solution_space, train_y, idx):
    cur_x = index_to_solution(train_solution_space, idx)
    return calculate_cost2(cur_x, train_y)  # n_sample x 1

def calculate_cost2(cur_x, train_y):
    if sum(train_y) == 0 or sum(train_y) == train_y.shape[0]:
        return np.ones(train_y.shape) / train_y.size
    clf = LogisticRegression().fit(cur_x,train_y)
    pred_y = clf.predict(cur_x)
    return 1*(pred_y == train_y) / train_y.size # n_sample x 1

def Competitive(train_solution_space, train_y, n_select=10):
    # train_solution_space: n_data[4289] x n_feature[4560]
    num_sol = 10
    num_data = train_y.shape[0]
    n_feature = train_solution_space.shape[1]
    n_internal = 25 # number of datas for calculating the internal fitness
    num_generations = 10

    num_parents_mating = num_sol//2
    num_parents_mating_competitive = n_internal//2
    num_mutations = 2
        
    new_population = generate_particle(num_sol)  # num_sol[5] x n_select[10]
    test_idx_y = np.random.randint(num_data, size=(n_internal,)) # n_internal[25] x 1

    idx = new_population[0,:]
    cur_cost = index_to_cost(train_solution_space, train_y, idx)
    best_solution = idx.copy()
    best_fitness = cur_cost.copy()

    start_time = timeit.default_timer()
    time_list = []
    for generation in range(num_generations):
        # [Fitness] Measuring the fitness of each chromosome in the population.
        # internal fitness
        fitness_grid = cal_fitness2(new_population, train_solution_space[test_idx_y,:], train_y[test_idx_y], num_sol)
        # print(fitness_grid.shape) # n_sol[5] x n_sample[25]
        fitness = np.sum(fitness_grid,1)
        if best_fitness < np.max(fitness):
            best_fitness = np.max(fitness)
            best_idx = np.argmax(fitness)
            best_solution = new_population[best_idx,:].copy()
            
        # print best and save time info.
        print("Generation %3d has Fitness=%.10f " % (generation+1, best_fitness),end='\r')
        time_list.append([timeit.default_timer()-start_time,best_fitness])

        ## Update the poputaion
        # [Select] Selecting the best parents in the population for mating
        select_idx = np.argsort(fitness)[-1:-num_parents_mating-1:-1]
        parents = new_population[select_idx,:]
        
        # [Crossover] Generating next generation using crossover.
        offspring_crossover = np.empty((num_sol-num_parents_mating,n_select),dtype='int')
        crossover_point = n_select//2 # The point crossover takes place between two parents	
        for k in range(num_sol-num_parents_mating):
            # Index of the first & second parent
            parent1_idx = k % num_parents_mating
            parent2_idx = (k+1) % num_parents_mating
            # single point crossover by parents
            offspring_crossover[k, :crossover_point] = parents[parent1_idx,:crossover_point]
            offspring_crossover[k, crossover_point:] = parents[parent2_idx,crossover_point:,]	

        # [Mutation] adding some variations to the offspring using mutation.
        mutations_counter = n_select // num_mutations
        for idx in range(num_sol-num_parents_mating):
            perturb_idx = np.random.randint(n_select)
            offspring_crossover[idx,perturb_idx] = (offspring_crossover[idx,perturb_idx] + np.random.randint(n_feature)) % n_feature
        

        # # [Select] Selecting the best parents in the population for mating
        # new_population
        fitness_competitive = np.sum(fitness_grid,0) # n_sample[25] x 1
        select_idx_competitive = np.argsort(fitness_competitive)[:num_parents_mating_competitive]
        parents_competitive = test_idx_y[select_idx_competitive]

        # [Mutation] adding some variations to the offspring using mutation.
        offspring_competitive = np.random.randint(num_data, size=(n_internal-num_parents_mating_competitive))

        # [Update] update the population
        new_population[:num_parents_mating,:] = parents
        new_population[num_parents_mating:,:] = offspring_crossover

        # [Update] upd ate the population
        test_idx_y[:num_parents_mating_competitive] = parents_competitive
        test_idx_y[num_parents_mating_competitive:] = offspring_competitive
    print('')
    return best_solution, time_list

def Cooperative(five_group_idx,train_solution_space, train_y, n_select=10):
    # train_solution_space: n_data[4289] x n_feature[4560]
    # G1 # F01 ~ F04 # two
    # G2 # F05 ~ F08 # two
    # G3 # F09 ~ F11 # two
    # G4 # F12 ~ F15 # two
    # G5 # F16 ~ F19 # two
    num_data = train_y.shape[0]
    n_feature = train_solution_space.shape[1]
    num_generations = 10

    
    n_group = 5
    num_sol = 5
    n_group_size = 2
    k = 5
    
    num_parents_mating = num_sol//2
    num_mutations = 2

    best_fitness = index_to_cost(train_solution_space, train_y, generate_particle(1)[0,:])

    # init
    new_population = np.zeros((num_sol,n_select),dtype='int')
    for i_col in range(num_sol):
        glob_concate_ary = [None for t in range(n_group)]
        for i_group in range(n_group):
                glob_concate_ary[i_group] = np.random.randint(five_group_idx[i_group].size, size=(1,2))
        new_population[i_col,:] = np.concatenate(glob_concate_ary,axis=1)


    start_time = timeit.default_timer()
    time_list = []
    for i_generation in range(num_generations):
        glob_concate_ary = [None for t in range(n_group)]
        for i_group in range(n_group):
            fit = np.zeros(num_sol)
            for i_sol in range(num_sol):
                concate_ary = [None for t in range(n_group)]
                concate_ary[i_group] = new_population[i_sol:i_sol+1,i_group*2:(i_group*2+2)]
                fitness_sum = 0
                for i_k in range(k): # k-fold test
                    for j_group in range(n_group):
                        if j_group != i_group:
                            concate_ary[j_group] = np.random.randint(five_group_idx[j_group].size, size=(1,2))
                    cur_sol = np.concatenate(concate_ary,axis=1)
                    cur_fitness = cal_fitness(cur_sol, train_solution_space, train_y, 1)
                    fitness_sum += cur_fitness
                fitness_sum /= n_group
                fit[i_sol] = fitness_sum

            select_idx = np.argsort(fit)[-1:-num_parents_mating-1:-1]
            new_population[:num_parents_mating,i_group*2:(i_group*2+2)] = new_population[select_idx,i_group*2:(i_group*2+2)]

            if fit[select_idx[0]] > best_fitness:
                best_fitness = fit[select_idx[0]]
            print("Generation %3d has Fitness=%.10f " % (i_generation+1, best_fitness),end='\r')
            time_list.append([timeit.default_timer()-start_time,best_fitness])

            for idx in range(num_sol-num_parents_mating):
                new_population[num_parents_mating+idx,i_group*2:(i_group*2+2)] = np.random.randint(five_group_idx[i_group].size, size=(1,2))
    print('')
    return new_population[0,:], time_list
        

def NSGA2(train_solution_space, train_y, n_select=10):
    # train_solution_space: n_data[4289] x n_feature[4560]
    # train_y: n_data[4289] x [3]
    num_sol = 20
    num_data = train_y.shape[0]
    n_feature = train_solution_space.shape[1]
    num_generations = 10

    num_parents_mating = num_sol//2
    num_mutations = 2
        
    new_population = generate_particle(num_sol)  # num_sol[5] x n_select[10]
    idx = new_population[0,:]
    best_solution = idx.copy()
    best_fitness1 = index_to_cost(train_solution_space, train_y[:,0], best_solution)
    best_fitness2 = index_to_cost(train_solution_space, train_y[:,1], best_solution)
    best_fitness3 = index_to_cost(train_solution_space, train_y[:,2], best_solution)

    for i_generation in range(num_generations):
        cur_fitness1 = cal_fitness(new_population, train_solution_space, train_y[:,0], num_sol)
        cur_fitness2 = cal_fitness(new_population, train_solution_space, train_y[:,1], num_sol)
        cur_fitness3 = cal_fitness(new_population, train_solution_space, train_y[:,2], num_sol)

        # sort by parato rank
        select_idx = []
        un_chosen_idx = [t for t in range(num_sol)]
        for i_rank in range(num_sol): # max rank is num_sol
            cur_front = []
            for i_sol in un_chosen_idx:
                new_front = []
                add_cur = True
                for i_prev_sol in range(len(cur_front)):
                    if cur_fitness1[cur_front[i_prev_sol]] > cur_fitness1[i_sol] and \
                        cur_fitness2[cur_front[i_prev_sol]] > cur_fitness2[i_sol] and \
                            cur_fitness3[cur_front[i_prev_sol]] > cur_fitness3[i_sol]:
                        new_front = cur_front
                        add_cur = False
                        break 
                    elif cur_fitness1[cur_front[i_prev_sol]] < cur_fitness1[i_sol] and \
                            cur_fitness2[cur_front[i_prev_sol]] < cur_fitness2[i_sol] and \
                                cur_fitness3[cur_front[i_prev_sol]] < cur_fitness3[i_sol]:
                        continue
                    else:
                        new_front.append(cur_front[i_prev_sol])
                if add_cur:
                    new_front.append(i_sol)
                cur_front = new_front.copy()
            # the solutions in current front exceed the number of archieve
            # choose by sparsity
            if len(select_idx) + len(cur_front) > num_parents_mating: 
                sparsity = np.zeros(len(cur_front))
                sort_ary = np.argsort(cur_fitness1[cur_front,0])
                for i_sort_ary in range(len(sort_ary)):
                    prev_idx = (i_sort_ary-1) if i_sort_ary > 0 else 0
                    next_idx = (i_sort_ary+1) if i_sort_ary < len(sort_ary)-1 else len(sort_ary)-1
                    sparsity[sort_ary[i_sort_ary]] += \
                        np.abs(cur_fitness1[sort_ary[prev_idx]] - cur_fitness1[sort_ary[next_idx]])
                sort_ary = np.argsort(cur_fitness2[cur_front,0])
                for i_sort_ary in range(len(sort_ary)):
                    prev_idx = (i_sort_ary-1) if i_sort_ary > 0 else 0
                    next_idx = (i_sort_ary+1) if i_sort_ary < len(sort_ary)-1 else len(sort_ary)-1
                    sparsity[sort_ary[i_sort_ary]] += \
                        np.abs(cur_fitness2[sort_ary[prev_idx]] - cur_fitness2[sort_ary[next_idx]])
                sort_ary = np.argsort(cur_fitness3[cur_front,0])
                for i_sort_ary in range(len(sort_ary)):
                    prev_idx = (i_sort_ary-1) if i_sort_ary > 0 else 0
                    next_idx = (i_sort_ary+1) if i_sort_ary < len(sort_ary)-1 else len(sort_ary)-1
                    sparsity[sort_ary[i_sort_ary]] += \
                        np.abs(cur_fitness3[sort_ary[prev_idx]] - cur_fitness3[sort_ary[next_idx]])
                n_remain = num_parents_mating-len(select_idx)
                sparse_idx = np.argsort(sparsity)[-1:-n_remain-1:-1]
                select_idx += [cur_front[t] for t in sparse_idx]
                break
            elif len(select_idx) + len(cur_front) == num_parents_mating: 
                select_idx += cur_front
                break
            select_idx += cur_front
            un_chosen_idx = [t for t in un_chosen_idx if (t not in cur_front)]
            if len(un_chosen_idx) == 0:
                break

        # [Select] Selecting the best parents in the population for mating
        parents = new_population[select_idx,:]
        best_solution = new_population[select_idx[0],:].copy()
        best_fitness1 = cur_fitness1[select_idx[0]]
        best_fitness2 = cur_fitness2[select_idx[0]]
        best_fitness3 = cur_fitness3[select_idx[0]]
        print("Generation %3d has Fitness=(%.10f,%.10f,%.10f) " % (i_generation+1, best_fitness1, best_fitness2, best_fitness3),end='\r')
        
        # [Crossover] Generating next generation using crossover.
        offspring_crossover = np.empty((num_sol-num_parents_mating,n_select),dtype='int')
        crossover_point = n_select//2 # The point crossover takes place between two parents	
        for k in range(num_sol-num_parents_mating):
            # Index of the first & second parent
            parent1_idx = k % num_parents_mating
            parent2_idx = (k+1) % num_parents_mating
            # single point crossover by parents
            offspring_crossover[k, :crossover_point] = parents[parent1_idx,:crossover_point]
            offspring_crossover[k, crossover_point:] = parents[parent2_idx,crossover_point:,]	

        # [Mutation] adding some variations to the offspring using mutation.
        mutations_counter = n_select // num_mutations
        for idx in range(num_sol-num_parents_mating):
            perturb_idx = np.random.randint(n_select)
            offspring_crossover[idx,perturb_idx] = (offspring_crossover[idx,perturb_idx] + np.random.randint(n_feature)) % n_feature
        
        # [Update] update the population
        new_population[:num_parents_mating,:] = parents
        new_population[num_parents_mating:,:] = offspring_crossover
        # print(new_population)

    print('')
    return best_solution

  

def GP(train_solution_space, train_y, n_tree_level=4, n_select = 10):
    # train_solution_space: n_data[4289] x n_feature[4560]
    # train_y: n_data[4289] x [5]
    inorder_seq = inorder_walk(0,[],n_tree_level)
    # operation_ary = ["+","-","*","+2*","+8*","+128*","+256*","+1024*","+0.5*","+0*","*0+","-128*"]
    operation_ary = ["+","-","*"]
    n_operation = len(operation_ary)
    n_tree_size = 2**n_tree_level - 1
    n_tree = 5
    n_data = train_y.shape[0]
    n_feature = train_solution_space.shape[1]
    n_class = 5
    n_generation = 40
    n_sol = 20
    n_parent = n_sol // 2
    num_mutations = n_sol // 3
    mutation_per_tree = n_tree_size // 4

    new_population = np.random.randint(n_operation, size=(n_tree,n_tree_size,n_sol))
    new_population[:,2**(n_tree_level-1)-1:,:] = np.random.randint(n_select, size=(n_tree,2**(n_tree_level-1),n_sol))
    x = np.random.randint(n_feature, size=(n_select,n_sol))
    
    best_fit = 0
    best_solution = None

    for i_generation in range(n_generation):
        tree_val = np.zeros((n_tree,n_data,n_sol))
        for i_sol in range(n_sol):
            for i_tree in range(n_tree):
                eval_str = ''
                for idx in inorder_seq:
                    if idx >= 2**(n_tree_level-1)-1: # the last level represents data
                        eval_str += ('train_solution_space[:,%d]' % x[new_population[i_tree,idx,i_sol],i_sol])
                    else: # otherwise, it is mapping to the operation array
                        eval_str += operation_ary[new_population[i_tree,idx,i_sol]]
                tree_val[i_tree,:,i_sol] = eval(eval_str)
        ## The tree sturcture is demonstrate below [attempt 1]:
        ## if tree_val[0,i_data] < 0:
        ##     if tree_val[1,i_data] < 0:
        ##         pred_y[i_data,3] = 1
        ##     else:
        ##         pred_y[i_data,4] = 1
        ## else:
        ##     if tree_val[2,i_data] < 0:
        ##         if tree_val[3,i_data] < 0:
        ##             pred_y[i_data,0] = 1
        ##         else:
        ##             pred_y[i_data,1] = 1
        ##     else:
        ##         if tree_val[4,i_data] < 0:
        ##             continue # all zeros
        ##         else:
        ##             pred_y[i_data,2] = 1
        pred_y = np.zeros((n_data,n_class,n_sol))
        fit = np.zeros((n_sol,n_class))
        for i_sol in range(n_sol):
            for i_class in range(n_class):
                pred_y[:,i_class,i_sol] = 1*(tree_val[i_class,:,i_sol] < 0)
                # clf = LogisticRegression().fit(tree_val[i_class,:,i_sol:i_sol+1],train_y[:,i_class:i_class+1])
                # pred_y[:,i_class,i_sol] = clf.predict(tree_val[i_class,:,i_sol:i_sol+1])
            fit[i_sol,:] = np.sum(1*(pred_y[:,:,i_sol] == train_y),0) / n_data # n_sample x 1
        mean_fit = np.mean(fit,1)
        if np.max(mean_fit) > best_fit:
            best_fit = np.max(mean_fit)
            best_idx = np.argmax(mean_fit)
            best_solution = new_population[:,:,best_idx].copy()
            for j_tree in range(n_tree):
                for j_node in range(2**(n_tree_level-1)-1,n_tree_size):
                    best_solution[j_tree,j_node] = x[best_solution[j_tree,j_node],best_idx]
        print("Generation %3d has mean Fitness=(%.10f) " % (i_generation+1,best_fit),end='\r')
        
        ## Update the poputaion
        # [Select] Selecting the best parents in the population for mating
        select_idx = np.argsort(mean_fit)[-1:-n_parent-1:-1]
        parents = new_population[:,:,select_idx] # n_tree x n_tree_size x n_parent
        
        # [Crossover] Generating next generation using crossover.
        offspring_crossover = np.empty((n_tree,n_tree_size,n_sol-n_parent),dtype='int')
        crossover_point = n_tree_size//2 # The point crossover takes place between two parents	
        for k in range(n_sol-n_parent):
            for i_tree in range(n_tree):
                # Index of the first & second parent
                parent1_idx = k % n_parent
                parent2_idx = (k+1) % n_parent
                # single point crossover by parents
                offspring_crossover[i_tree, :crossover_point, k] = parents[i_tree,:crossover_point,parent1_idx]
                offspring_crossover[i_tree, crossover_point:, k] = parents[i_tree,crossover_point:,parent2_idx]	
        # [Mutation]
        for idx in range(num_mutations):
            for _ in range(mutation_per_tree):
                perturb_idx = np.random.randint(2**(n_tree_level-1) - 1)
                for i_tree in range(n_tree):
                    offspring_crossover[i_tree,perturb_idx,-idx] = np.random.randint(n_operation)
            perturb_idx = np.random.randint(2**(n_tree_level-1)) + 2**(n_tree_level-1) - 1
            for i_tree in range(n_tree):
                offspring_crossover[i_tree,perturb_idx,-idx] = np.random.randint(n_select)
        
        # [Mutation]
        x[:,-num_mutations:] = np.random.randint(n_feature, size=(n_select,num_mutations)) # n_select x n_sol

        # [Update]
        new_population[:,:,:n_parent] = parents
        new_population[:,:,n_parent:] = offspring_crossover
        # print('------------------------')
        # print(new_population[0,:,0])
        # print(new_population[1,:,0])
        # print(new_population[2,:,0])
        # print(new_population[3,:,0])
        # print(new_population[4,:,0])
        # print('------------------------')
        # print(new_population[0,:,:])
    print('')
    return best_solution

