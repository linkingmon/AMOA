import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from utils.preprocess import get_data, get_solution_space, inorder_walk
from utils.eval import get_score
from utils.optimization import GP, Competitive, Cooperative, NSGA2, index_to_solution
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
n_select = 10 # Number of selection of bio-markers

train = pd.read_excel ('./Data.xlsx').to_numpy()
valid = pd.read_excel ('./Test1.xlsx').to_numpy()

print('=============== Read data and preprocess ===============')
train_x, train_y = get_data(train)
valid_x, valid_y = get_data(valid)
train_solution_space, _ = get_solution_space(train_x)
valid_solution_space, _ = get_solution_space(valid_x)
print(train_solution_space.shape)

print('===========     Run Genetic Programming      ==========')
idx_GP = np.load(open('model/idx_GP'+seed_name+'.npy', 'rb'))
print(idx_GP.shape)
n_tree_level = 4
n_tree = 5
operation_ary = ["+","-","*"]
inorder_seq = inorder_walk(0,[],n_tree_level)
pred_y_GP = np.zeros((train_x.shape[0],n_y))
pred_y_GP_valid = np.zeros((valid_x.shape[0],n_y))
for i_tree in range(n_tree):
    eval_str = ''
    for idx in inorder_seq:
        if idx >= 2**(n_tree_level-1)-1: # the last level represents data
            eval_str += ('train_solution_space[:,%d]' % idx_GP[i_tree,idx])
        else: # otherwise, it is mapping to the operation array
            eval_str += operation_ary[idx_GP[i_tree,idx]]
    pred_y_GP[:,i_tree] = 1*(eval(eval_str) < 0)  
    eval_str = ''
    for idx in inorder_seq:
        if idx >= 2**(n_tree_level-1)-1: # the last level represents data
            eval_str += ('valid_solution_space[:,%d]' % idx_GP[i_tree,idx])
        else: # otherwise, it is mapping to the operation array
            eval_str += operation_ary[idx_GP[i_tree,idx]]
    print(eval_str)
    pred_y_GP_valid[:,i_tree] = 1*(eval(eval_str) < 0)  

print('===========    Run Competivie Coevolution    ==========')
pred_y_Competitive = np.zeros((train_x.shape[0],n_y))
pred_y_Competitive_valid = np.zeros((valid_x.shape[0],n_y))
idx_Competitive = np.load(open('model/idx_Competitive'+seed_name+'.npy', 'rb'))
print(idx_Competitive.shape)
for i_class in range(n_y):
    print(idx_Competitive[:,i_class])
    cur_x =  index_to_solution(train_solution_space, idx_Competitive[:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_Competitive[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_Competitive[:,i_class])
    pred_y_Competitive_valid[:,i_class] = clf.predict(cur_x)
    
print('===========   Run Cooperative Coevolution    ==========')
pred_y_Cooperative = np.zeros((train_x.shape[0],n_y))
pred_y_Cooperative_valid = np.zeros((valid_x.shape[0],n_y))
idx_Cooperative = np.load(open('model/idx_Cooperative'+seed_name+'.npy', 'rb'))
print(idx_Cooperative.shape)
for i_class in range(n_y):
    print(idx_Cooperative[:,i_class])
    cur_x =  index_to_solution(train_solution_space, idx_Cooperative[:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_Cooperative[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_Cooperative[:,i_class])
    pred_y_Cooperative_valid[:,i_class] = clf.predict(cur_x)
        
print('===========            Run NSGA-II           ==========')
pred_y_NSGA2 = np.zeros((train_x.shape[0],n_y))
pred_y_NSGA2_valid = np.zeros((valid_x.shape[0],n_y))
idx_NSGA2 = np.load(open('model/idx_NSGA2'+seed_name+'.npy', 'rb'))
print(idx_NSGA2.shape)
print(idx_NSGA2)
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_NSGA2)
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_NSGA2[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(valid_solution_space, idx_NSGA2)
    pred_y_NSGA2_valid[:,i_class] = clf.predict(cur_x)

# Evaluate output
score_GP, _ = get_score(train_y,pred_y_GP)
score_Competitive, _ = get_score(train_y,pred_y_Competitive)
score_Cooperative, _ = get_score(train_y,pred_y_Cooperative)
score_NSGA2, _ = get_score(train_y,pred_y_NSGA2)

num_GP = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_GP]
num_Competitive = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_Competitive]
num_Cooperative = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_Cooperative]
num_NSGA2 = [int(train_x.shape[0]-t*train_x.shape[0]) for t in score_NSGA2]

score_GP_valid, _ = get_score(valid_y,pred_y_GP_valid)
score_Competitive_valid, _ = get_score(valid_y,pred_y_Competitive_valid)
score_Cooperative_valid, _ = get_score(valid_y,pred_y_Cooperative_valid)
score_NSGA2_valid, _ = get_score(valid_y,pred_y_NSGA2_valid)

num_GP_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_GP_valid]
num_Competitive_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_Competitive_valid]
num_Cooperative_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_Cooperative_valid]
num_NSGA2_valid = [int(valid_x.shape[0]-t*valid_x.shape[0]) for t in score_NSGA2_valid]

print('============== Print Algorithm Results ================')
table = [['GP']+score_GP+[np.mean(score_GP)], \
         ['Competitive']+score_Competitive+[np.mean(score_Competitive)], \
         ['Cooperative']+score_Cooperative+[np.mean(score_Cooperative)], \
         ['NSGA2']+score_NSGA2+[np.mean(score_NSGA2)]]
table_valid = [['GP']+score_GP_valid+[np.mean(score_GP_valid)], \
               ['Competitive']+score_Competitive_valid+[np.mean(score_Competitive_valid)], \
               ['Cooperative']+score_Cooperative_valid+[np.mean(score_Cooperative_valid)], \
               ['NSGA2']+score_NSGA2_valid+[np.mean(score_NSGA2_valid)]]
table_num = [['GP']+num_GP, \
             ['Competitive']+num_Competitive, \
             ['Cooperative']+num_Cooperative, \
             ['NSGA2']+num_NSGA2]
table_num_valid = [['GP']+num_GP_valid, \
                   ['Competitive']+num_Competitive_valid, \
                   ['Cooperative']+num_Cooperative_valid, \
                   ['NSGA2']+num_NSGA2_valid]
print('GP: Genetic Programming')
print('Competitive: Competitive Coevolution')
print('Cooperative: Cooperative Coevolution')
print('NSGA2')
print('>> Training performance')
print(tabulate(table, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Mean Acc.'], tablefmt='orgtbl'))
print('>> Validation performance')
print(tabulate(table_valid, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Mean Acc.'], tablefmt='orgtbl'))
print('>> Training performance #number')
print(tabulate(table_num, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], tablefmt='orgtbl'))
print('>> Validation performance #number')
print(tabulate(table_num_valid, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], tablefmt='orgtbl'))

score = np.array([score_Competitive, score_Cooperative, score_NSGA2])