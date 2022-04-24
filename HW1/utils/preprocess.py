from cmath import nan
import numpy as np
from scipy.stats import pearsonr

def get_data(train, n_f=19, n_r=5, n_y=5):
    print('Remove Data with Missing Values ...')
    unremove_idx = (1-np.sum(np.isnan(train),axis=1)).nonzero()[0] # remove the data with missing values
    train_f = train[unremove_idx,:n_f]
    train_r = train[unremove_idx,n_f:n_f+n_r]
    train_y = train[unremove_idx,n_f+n_r:]

    train_x = (np.tile(train_f,(1,n_r)) / np.repeat(train_r,n_f,axis=1))
    print('Finish data read, x & y are with size ...',train_x.shape,train_y.shape)
    return train_x, train_y

def get_solution_space(train_x):
    n_sample = train_x.shape[0]
    # n_sample = 10
    n_feature = train_x.shape[1]
    solution_space = np.zeros((train_x.shape[0],n_feature,n_feature))
    for i_sample in range(n_sample):
        print('Generating the solution spaces ... (%d/%d)' % (i_sample, train_x.shape[0]), end='\r')
        for i_col in range(n_feature):
            for i_row in range(n_feature):
                if i_col == i_row:
                    solution_space[i_sample][i_col][i_row] = train_x[i_sample,i_col]
                else:
                    solution_space[i_sample][i_col][i_row] = train_x[i_sample,i_col]*train_x[i_sample,i_row]
    print('')
    return solution_space
