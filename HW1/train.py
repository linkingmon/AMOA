import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate
import timeit
import pickle

import argparse


from utils.preprocess import get_data, get_solution_space
from utils.eval import get_score
from utils.optimization import Ant_colony, Differential_evolution, Particle_swarm, Simulated_Annealing, Genetic_algorithm, index_to_solution

## Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='the random seed of the code', default=0, type=int)
args = parser.parse_args()


np.random.seed(args.seed)
seed_name = '' if args.seed == 0 else ('_seed%d' % args.seed)

n_f = 19      # Number of F type bio-markers
n_r = 5       # Number of R type bio-markers
n_y = 5       # Number of prediction classes
n_select = 30 # Number of selection of bio-markers

train = pd.read_excel ('./Data.xlsx').to_numpy()
valid = pd.read_excel ('./Test1.xlsx').to_numpy()
test = pd.read_excel ('./Test2.xlsx').to_numpy()

print('=============== Read data and preprocess ===============')
train_x, train_y = get_data(train)
train_solution_space = get_solution_space(train_x)
print(train_solution_space.shape)

start_SA = timeit.default_timer()
print('=========== Run Simulated Anneling Algorithm ===========')
idx_SA = np.zeros((2,n_select,n_y),dtype='int')
pred_y_SA = np.zeros((train_x.shape[0],n_y))
time_list_SA = []
for i_class in range(n_y):
    idx_SA[:,:,i_class],time_list = Simulated_Annealing(train_solution_space, train_y[:,i_class])
    time_list_SA.append(time_list)
    cur_x =  index_to_solution(train_solution_space, idx_SA[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_SA[:,i_class] = clf.predict(cur_x)
stop_SA = timeit.default_timer()
np.save('model/idx_SA'+seed_name+'.npy',idx_SA)
print('Save results to model/idx_SA.npy ...')
with open("time/SA.list", "wb") as fp:
    pickle.dump(time_list_SA, fp)

start_GA = timeit.default_timer()
print('================= Run Genetic Algorithm ================')
idx_GA = np.zeros((2,n_select,n_y),dtype='int')
pred_y_GA = np.zeros((train_x.shape[0],n_y))
time_list_GA = []
for i_class in range(n_y):
    idx_GA[:,:,i_class],time_list = Genetic_algorithm(train_solution_space, train_y[:,i_class])
    time_list_GA.append(time_list)
    cur_x =  index_to_solution(train_solution_space, idx_GA[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_GA[:,i_class] = clf.predict(cur_x)
stop_GA = timeit.default_timer()
np.save('model/idx_GA'+seed_name+'.npy',idx_GA)
print('Save results to model/idx_GA.npy ...')
with open("time/GA.list", "wb") as fp:
    pickle.dump(time_list_GA, fp)

start_PS = timeit.default_timer()
print('================= Run Particle Swarm ==================')
idx_PS = np.zeros((2,n_select,n_y),dtype='int')
pred_y_PS = np.zeros((train_x.shape[0],n_y))
time_list_PS = []
for i_class in range(n_y):
    idx_PS[:,:,i_class],time_list = Particle_swarm(train_solution_space, train_y[:,i_class])
    time_list_PS.append(time_list)
    cur_x =  index_to_solution(train_solution_space, idx_PS[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_PS[:,i_class] = clf.predict(cur_x)
stop_PS = timeit.default_timer()
np.save('model/idx_PS'+seed_name+'.npy',idx_PS)
print('Save results to model/idx_PS.npy ...')
with open("time/PS.list", "wb") as fp:
    pickle.dump(time_list_PS, fp)

start_AC = timeit.default_timer()
print('=================== Run Ant Colony ====================')
idx_AC = np.zeros((2,n_select,n_y),dtype='int')
pred_y_AC = np.zeros((train_x.shape[0],n_y))
time_list_AC = []
for i_class in range(n_y):
    idx_AC[:,:,i_class],time_list = Ant_colony(train_solution_space, train_y[:,i_class])
    time_list_AC.append(time_list)
    cur_x =  index_to_solution(train_solution_space, idx_AC[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_AC[:,i_class] = clf.predict(cur_x)
stop_AC = timeit.default_timer()
np.save('model/idx_AC'+seed_name+'.npy',idx_AC)
print('Save results to model/idx_AC.npy ...')
with open("time/AC.list", "wb") as fp:
    pickle.dump(time_list_AC, fp)

start_DE = timeit.default_timer()
print('============= Run Differential Evolution ==============')
idx_DE = np.zeros((2,n_select,n_y),dtype='int')
pred_y_DE = np.zeros((train_x.shape[0],n_y))
time_list_DE = []
for i_class in range(n_y):
    idx_DE[:,:,i_class],time_list = Differential_evolution(train_solution_space, train_y[:,i_class])
    time_list_DE.append(time_list)
    cur_x =  index_to_solution(train_solution_space, idx_DE[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_DE[:,i_class] = clf.predict(cur_x)
stop_DE = timeit.default_timer()
np.save('model/idx_DE'+seed_name+'.npy',idx_DE)
print('Save results to model/idx_DE.npy ...')
with open("time/DE.list", "wb") as fp:
    pickle.dump(time_list_DE, fp)

print('=================== Output Run Time ===================')
print('SA Time: ', stop_SA - start_SA)  
print('GA Time: ', stop_GA - start_GA)  
print('PS Time: ', stop_PS - start_PS)  
print('AC Time: ', stop_AC - start_AC)  
print('DE Time: ', stop_DE - start_DE)  