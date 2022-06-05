import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tabulate import tabulate

from utils.preprocess import get_data, get_solution_space
from utils.optimization import GP, Competitive, Cooperative, NSGA2, index_to_solution

n_f = 19      # Number of F type bio-markers
n_r = 5       # Number of R type bio-markers
n_y = 5       # Number of prediction classes
n_select = 10 # Number of selection of bio-markers

train = pd.read_excel ('./Data.xlsx').to_numpy()
df = pd.read_excel ('./Test2.xlsx')
test = pd.read_excel ('./Test2.xlsx').to_numpy()

print('=============== Read data and preprocess ===============')
train_x, train_y = get_data(train)
test_x, test_y = get_data(test)
train_solution_space, _ = get_solution_space(train_x)
test_solution_space, _ = get_solution_space(test_x)
print(train_solution_space.shape)

print('===========   Run Cooperative Coevolution    ==========')
pred_y_Cooperative = np.zeros((train_x.shape[0],n_y))
pred_y_Cooperative_test = np.zeros((test_x.shape[0],n_y))
idx_Cooperative = np.load(open('model/idx_Cooperative.npy', 'rb'))
for i_class in range(n_y):
    cur_x =  index_to_solution(train_solution_space, idx_Cooperative[:,i_class])
    clf = LogisticRegression().fit(cur_x,train_y[:,i_class])
    pred_y_Cooperative[:,i_class] = clf.predict(cur_x)
    cur_x =  index_to_solution(test_solution_space, idx_Cooperative[:,i_class])
    pred_y_Cooperative_test[:,i_class] = clf.predict(cur_x)


df['C01'][:] = pred_y_Cooperative_test[:,0]
df['C02'][:] = pred_y_Cooperative_test[:,1]
df['C03'][:] = pred_y_Cooperative_test[:,2]
df['C04'][:] = pred_y_Cooperative_test[:,3]
df['C05'][:] = pred_y_Cooperative_test[:,4]


df.to_excel("Test2_Answer.xlsx", sheet_name='Sheet_1')  