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

def get_nonlinear_data(train_x):
    n_feature = train_x.shape[1]
    print('Generating sqaure term features ...')
    train_x_square = np.zeros((train_x.shape[0],n_feature*(n_feature-1)//2))
    cnt = 0
    for i_feature in range(n_feature):
        for j_feature in range(i_feature+1, n_feature):
            train_x_square[:,cnt] = train_x[:,i_feature]*train_x[:,j_feature]
            cnt += 1
    # print('Generating cubic term features ...')
    # train_x_cubic = np.zeros((train_x.shape[0],n_feature*(n_feature-1)*(n_feature-2)//6))
    # cnt = 0
    # for i_feature in range(n_feature):
    #     for j_feature in range(i_feature+1, n_feature):
    #         for k_feature in range(j_feature+1, n_feature):
    #             train_x_cubic[:,cnt] = train_x[:,i_feature]*train_x[:,j_feature]*train_x[:,k_feature]
    #             cnt += 1
    # train_x = np.hstack((train_x,train_x_square,train_x_cubic))
    # print('Augmented Data has size', train_x.shape)
    train_x = np.hstack((train_x,train_x_square))
    return train_x

def select_idx(train_x, train_y, algor, n_select=30, n_y=5):
    idx = np.zeros((n_select, n_y), dtype=int)
    n_feature = train_x.shape[1]
    for i_y in range(n_y):
        pred = train_y[:,i_y]*2-1 # convert 0, 1 to -1, 1
        corr = np.zeros(n_feature)
        for i_feature in range(n_feature):
            corr[i_feature], _ = np.abs(pearsonr(train_x[:,i_feature],pred))
        idx[:,i_y] = np.argsort(corr)[-1:-n_select-1:-1]
    return idx
