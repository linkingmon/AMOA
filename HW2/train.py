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
from utils.optimization import GP, Competitive, Cooperative, NSGA2, index_to_solution

## Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--seed', help='the random seed of the code', default=0, type=int)
args = parser.parse_args()


np.random.seed(args.seed)
seed_name = '' if args.seed == 0 else ('_seed%d' % args.seed)

n_f = 19      # Number of F type bio-markers
n_r = 5       # Number of R type bio-markers
n_y = 5       # Number of prediction classes
n_select = 10 # Number of selection of bio-markers

train = pd.read_excel ('./Data.xlsx').to_numpy()
valid = pd.read_excel ('./Test1.xlsx').to_numpy()
test = pd.read_excel ('./Test2.xlsx').to_numpy()

print('=============== Read data and preprocess ===============')
train_x, train_y = get_data(train)
train_solution_space, five_group_idx = get_solution_space(train_x)
print(train_solution_space.shape)
print(five_group_idx[0].shape)
print(five_group_idx[1].shape)
print(five_group_idx[2].shape)
print(five_group_idx[3].shape)
print(five_group_idx[4].shape)

print('===========     Run Genetic Programming      ==========')
n_tree_level = 3
n_tree_size = 2**n_tree_level - 1
idx_GP = np.zeros((n_tree_size,n_y),dtype='int')
idx_GP = GP(train_solution_space, train_y)
print(idx_GP)
np.save('model/idx_GP'+seed_name+'.npy',idx_GP)
print('Save results to model/idx_GP.npy ...')


print('===========    Run Competivie Coevolution    ==========')
idx_Competitive = np.zeros((n_select,n_y),dtype='int')
pred_y_Competitive = np.zeros((train_x.shape[0],n_y))
for i_class in range(n_y):
    idx_Competitive[:,i_class],_ = Competitive(train_solution_space, train_y[:,i_class])
    print("CLASS",i_class, idx_Competitive[:,i_class])
    cur_x =  index_to_solution(train_solution_space, idx_Competitive[:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_Competitive[:,i_class] = clf.predict(cur_x)
np.save('model/idx_Competitive'+seed_name+'.npy',idx_Competitive)
print('Save results to model/idx_Competitive.npy ...')
print(get_score(train_y, pred_y_Competitive, n_y=5))

print('===========   Run Cooperative Coevolution    ==========')
idx_Cooperative = np.zeros((n_select,n_y),dtype='int')
pred_y_Cooperative = np.zeros((train_x.shape[0],n_y))
for i_class in range(n_y):
    idx_Cooperative[:,i_class],_ = Cooperative(five_group_idx,train_solution_space, train_y[:,i_class])
    print("CLASS",i_class, idx_Cooperative[:,i_class])
    cur_x =  index_to_solution(train_solution_space, idx_Cooperative[:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_Cooperative[:,i_class] = clf.predict(cur_x)
np.save('model/idx_Cooperative'+seed_name+'.npy',idx_Cooperative)
print('Save results to model/idx_Cooperative.npy ...')
print(get_score(train_y, pred_y_Cooperative, n_y=5))

print('===========            Run NSGA-II           ==========')
idx_NSGA2 = np.zeros((n_select),dtype='int')
pred_y_NSGA2 = np.zeros((train_x.shape[0],n_y))
idx_NSGA2 = NSGA2(train_solution_space, train_y[:,[0,1,3]])
for i_class in [0, 1, 3]:
    print("CLASS",i_class, idx_NSGA2)
    cur_x =  index_to_solution(train_solution_space, idx_NSGA2)
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_NSGA2[:,i_class] = clf.predict(cur_x)
np.save('model/idx_NSGA2'+seed_name+'.npy',idx_NSGA2)
print('Save results to model/idx_NSGA2.npy ...')
print(get_score(train_y, pred_y_NSGA2, n_y=5))
