import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tabulate import tabulate

from utils.preprocess import get_data, select_idx, get_nonlinear_data
from utils.eval import get_score
from utils.optimization import Ant_colony, Differential_evolution, Particle_swarm, Simulated_Annealing, Genetic_algorithm, predict

n_f = 19   # Number of F type bio-markers
n_r = 5    # Number of R type bio-markers
n_y = 5    # Number of prediction classes

train = pd.read_excel ('./Data.xlsx').to_numpy()
valid = pd.read_excel ('./Test1.xlsx').to_numpy()
test = pd.read_excel ('./Test2.xlsx').to_numpy()

print('=============== Read data and preprocess ===============')
train_x, train_y = get_data(train)
valid_x, valid_y = get_data(valid)
test_x, _ = get_data(test)
train_x = get_nonlinear_data(train_x)
valid_x = get_nonlinear_data(valid_x)
test_x = get_nonlinear_data(test_x)
idx = select_idx(train_x, train_y, 'Max_Corr')
train_x = train_x[:,idx]
valid_x = valid_x[:,idx]
test_x = test_x[:,idx]
print(train_x.shape)
    
print('=========== Run Simulated Anneling Algorithm ===========')
coeff_SA = np.zeros((train_x.shape[1],n_y))
pred_y_SA = np.zeros(train_y.shape)
pred_y_SA_valid = np.zeros(valid_y.shape)
for i_class in range(n_y):
    coeff_SA[:,i_class:i_class+1] = Simulated_Annealing(train_x[:,:,i_class], train_y[:,i_class:i_class+1])
    pred_y_SA[:,i_class:i_class+1] = predict(train_x[:,:,i_class],coeff_SA[:,i_class:i_class+1])
    pred_y_SA_valid[:,i_class:i_class+1] = predict(valid_x[:,:,i_class],coeff_SA[:,i_class:i_class+1])

print('================= Run Genetic Algorithm ================')
coeff_GA = np.zeros((train_x.shape[1],n_y))
pred_y_GA = np.zeros(train_y.shape)
pred_y_GA_valid = np.zeros(valid_y.shape)
for i_class in range(n_y):
    coeff_GA[:,i_class:i_class+1] = Genetic_algorithm(train_x[:,:,i_class], train_y[:,i_class:i_class+1])
    pred_y_GA[:,i_class:i_class+1] = predict(train_x[:,:,i_class],coeff_GA[:,i_class:i_class+1])
    pred_y_GA_valid[:,i_class:i_class+1] = predict(valid_x[:,:,i_class],coeff_GA[:,i_class:i_class+1])

print('================= Run Particle Swarm ==================')
coeff_PS = np.zeros((train_x.shape[1],n_y))
pred_y_PS = np.zeros(train_y.shape)
pred_y_PS_valid = np.zeros(valid_y.shape)
for i_class in range(n_y):
    coeff_PS[:,i_class:i_class+1] = Particle_swarm(train_x[:,:,i_class], train_y[:,i_class:i_class+1])
    pred_y_PS[:,i_class:i_class+1] = predict(train_x[:,:,i_class],coeff_PS[:,i_class:i_class+1])
    pred_y_PS_valid[:,i_class:i_class+1] = predict(valid_x[:,:,i_class],coeff_PS[:,i_class:i_class+1])

print('=================== Run Ant Colony ====================')
coeff_AC = np.zeros((train_x.shape[1],n_y))
pred_y_AC = np.zeros(train_y.shape)
pred_y_AC_valid = np.zeros(valid_y.shape)
for i_class in range(n_y):
    coeff_AC[:,i_class:i_class+1] = Ant_colony(train_x[:,:,i_class], train_y[:,i_class:i_class+1])
    pred_y_AC[:,i_class:i_class+1] = predict(train_x[:,:,i_class],coeff_AC[:,i_class:i_class+1])
    pred_y_AC_valid[:,i_class:i_class+1] = predict(valid_x[:,:,i_class],coeff_AC[:,i_class:i_class+1])

print('============= Run Differential Evolution ==============')
coeff_DE = np.zeros((train_x.shape[1],n_y))
pred_y_DE = np.zeros(train_y.shape)
pred_y_DE_valid = np.zeros(valid_y.shape)
for i_class in range(n_y):
    coeff_DE[:,i_class:i_class+1] = Differential_evolution(train_x[:,:,i_class], train_y[:,i_class:i_class+1])
    pred_y_DE[:,i_class:i_class+1] = predict(train_x[:,:,i_class],coeff_DE[:,i_class:i_class+1])
    pred_y_DE_valid[:,i_class:i_class+1] = predict(valid_x[:,:,i_class],coeff_DE[:,i_class:i_class+1])

# Evaluate output
score_SA, _ = get_score(train_y,pred_y_SA)
score_GA, _ = get_score(train_y,pred_y_GA)
score_PS, _ = get_score(train_y,pred_y_PS)
score_AC, _ = get_score(train_y,pred_y_AC)
score_DE, _ = get_score(train_y,pred_y_DE)

score_SA_valid, _ = get_score(valid_y,pred_y_SA_valid)
score_GA_valid, _ = get_score(valid_y,pred_y_GA_valid)
score_PS_valid, _ = get_score(valid_y,pred_y_PS_valid)
score_AC_valid, _ = get_score(valid_y,pred_y_AC_valid)
score_DE_valid, _ = get_score(valid_y,pred_y_DE_valid)

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
print('SA: Simulated Anneling')
print('GA: Genetic Algorithm')
print('PS: Particle Swarm')
print('AC: Ant Colony')
print('DE: Differential Evolution')
print('>> Training performance')
print(tabulate(table, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Mean Acc.'], tablefmt='orgtbl'))
print('>> Validation performance')
print(tabulate(table_valid, headers=['Algor.', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Mean Acc.'], tablefmt='orgtbl'))

print('================= Output Test2 by DE ==================')
