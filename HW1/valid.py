import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from utils.preprocess import get_data, get_solution_space
from utils.eval import get_score
from utils.optimization import Ant_colony, Differential_evolution, Particle_swarm, Simulated_Annealing, Genetic_algorithm, index_to_solution
import argparse

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

print('=============== Read data and preprocess ===============')
train_x, train_y = get_data(train)
valid_x, valid_y = get_data(valid)
train_solution_space = get_solution_space(train_x)
valid_solution_space = get_solution_space(valid_x)
print(train_solution_space.shape)

print('=========== Run Simulated Anneling Algorithm ===========')
idx_SA = np.zeros((2,n_select,n_y),dtype='int')
pred_y_SA = np.zeros((train_x.shape[0],n_y))
pred_y_SA_valid = np.zeros((valid_x.shape[0],n_y))
idx_SA = np.load(open('model/idx_SA'+seed_name+'.npy', 'rb'))
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_SA[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_SA[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_SA[:,:,i_class])
    pred_y_SA_valid[:,i_class] = clf.predict(cur_x)

print('================= Run Genetic Algorithm ================')
idx_GA = np.zeros((2,n_select,n_y),dtype='int')
pred_y_GA = np.zeros((train_x.shape[0],n_y))
pred_y_GA_valid = np.zeros((valid_x.shape[0],n_y))
idx_GA = np.load(open('model/idx_GA'+seed_name+'.npy', 'rb'))
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_GA[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_GA[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_GA[:,:,i_class])
    pred_y_GA_valid[:,i_class] = clf.predict(cur_x)

print('================= Run Particle Swarm ==================')
idx_PS = np.zeros((2,n_select,n_y),dtype='int')
pred_y_PS = np.zeros((train_x.shape[0],n_y))
pred_y_PS_valid = np.zeros((valid_x.shape[0],n_y))
idx_PS = np.load(open('model/idx_PS'+seed_name+'.npy', 'rb'))
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_PS[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_PS[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_PS[:,:,i_class])
    pred_y_PS_valid[:,i_class] = clf.predict(cur_x)
score_PS, _ = get_score(train_y,pred_y_PS)

print('=================== Run Ant Colony ====================')
idx_AC = np.zeros((2,n_select,n_y),dtype='int')
pred_y_AC = np.zeros((train_x.shape[0],n_y))
pred_y_AC_valid = np.zeros((valid_x.shape[0],n_y))
idx_AC = np.load(open('model/idx_AC'+seed_name+'.npy', 'rb'))
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_AC[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_AC[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_AC[:,:,i_class])
    pred_y_AC_valid[:,i_class] = clf.predict(cur_x)

print('============= Run Differential Evolution ==============')
idx_DE = np.zeros((2,n_select,n_y),dtype='int')
pred_y_DE = np.zeros((train_x.shape[0],n_y))
pred_y_DE_valid = np.zeros((valid_x.shape[0],n_y))
idx_DE = np.load(open('model/idx_DE'+seed_name+'.npy', 'rb'))
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_DE[:,:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_DE[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_DE[:,:,i_class])
    pred_y_DE_valid[:,i_class] = clf.predict(cur_x)


# Evaluate output
score_SA, _ = get_score(train_y,pred_y_SA)
score_GA, _ = get_score(train_y,pred_y_GA)
score_PS, _ = get_score(train_y,pred_y_PS)
score_AC, _ = get_score(train_y,pred_y_AC)
score_DE, _ = get_score(train_y,pred_y_DE)
num_SA = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_SA]
num_GA = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_GA]
num_PS = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_PS]
num_AC = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_AC]
num_DE = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_DE]

score_SA_valid, _ = get_score(valid_y,pred_y_SA_valid)
score_GA_valid, _ = get_score(valid_y,pred_y_GA_valid)
score_PS_valid, _ = get_score(valid_y,pred_y_PS_valid)
score_AC_valid, _ = get_score(valid_y,pred_y_AC_valid)
score_DE_valid, _ = get_score(valid_y,pred_y_DE_valid)
num_SA_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_SA_valid]
num_GA_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_GA_valid]
num_PS_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_PS_valid]
num_AC_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_AC_valid]
num_DE_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_DE_valid]

print('============== Print Algorithm Results ================')
table = [['SA']+score_SA+[np.mean(score_SA)], \
         ['GA']+score_GA+[np.mean(score_GA)], \
         ['PS']+score_PS+[np.mean(score_PS)], \
         ['AC']+score_AC+[np.mean(score_AC)], \
         ['DE']+score_DE+[np.mean(score_DE)]]
table_valid = [['SA']+score_SA_valid+[np.mean(score_SA_valid)], \
               ['GA']+score_GA_valid+[np.mean(score_GA_valid)], \
               ['PS']+score_PS_valid+[np.mean(score_PS_valid)], \
               ['AC']+score_AC_valid+[np.mean(score_AC_valid)], \
               ['DE']+score_DE_valid+[np.mean(score_DE_valid)]]
table_num = [['SA']+num_SA, \
             ['GA']+num_GA, \
             ['PS']+num_PS, \
             ['AC']+num_AC, \
             ['DE']+num_DE]
table_num_valid = [['SA']+num_SA_valid, \
                   ['GA']+num_GA_valid, \
                   ['PS']+num_PS_valid, \
                   ['AC']+num_AC_valid, \
                   ['DE']+num_DE_valid]
print('SA: Simulated Anneling')
print('GA: Genetic Algorithm')
print('PS: Particle Swarm')
print('AC: Ant Colony')
print('DE: Differential Evolution')
print('>> Training performance')
print(tabulate(table, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Mean Acc.'], tablefmt='orgtbl'))
print('>> Validation performance')
print(tabulate(table_valid, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Mean Acc.'], tablefmt='orgtbl'))
print('>> Training performance #number')
print(tabulate(table_num, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], tablefmt='orgtbl'))
print('>> Validation performance #number')
print(tabulate(table_num_valid, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], tablefmt='orgtbl'))

score = np.array([score_SA, score_GA, score_PS, score_AC, score_DE])
np.save('acc/score'+seed_name+'.npy',score)