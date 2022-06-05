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

def get_solution_space(train_x,n_f=19):
    n_sample = train_x.shape[0]
    # n_sample = 10
    n_feature = train_x.shape[1]
    solution_space = np.zeros((train_x.shape[0],n_feature+n_feature*(n_feature-1)//2))
    five_group_idx = [[],[],[],[],[]]
    
    for i_col in range(n_feature):
        if i_col % n_f <= 3:
            five_group_idx[0].append(i_col)
        elif i_col % n_f <= 7:
            five_group_idx[1].append(i_col)
        elif i_col % n_f <= 10:
            five_group_idx[2].append(i_col)
        elif i_col % n_f <= 14:
            five_group_idx[3].append(i_col)
        elif i_col % n_f <= 18:
            five_group_idx[4].append(i_col)
    for i_sample in range(n_sample):
        print('Generating the solution spaces ... (%d/%d)' % (i_sample, train_x.shape[0]), end='\r')
        for i_col in range(n_feature):
            solution_space[i_sample][i_col] = train_x[i_sample,i_col]
    
    for i_sample in range(n_sample):
        print('Generating the solution spaces ... (%d/%d)' % (i_sample, train_x.shape[0]), end='\r')
        cnt = n_feature
        for i_col in range(n_feature):
            for i_row in range(i_col+1,n_feature):
                solution_space[i_sample][cnt] = train_x[i_sample,i_col]*train_x[i_sample,i_row]
                cnt += 1

                
    cnt = n_feature
    for i_col in range(n_feature):
        for i_row in range(i_col+1,n_feature):
            if i_col % n_f <= 3:
                five_group_idx[0].append(cnt)
            elif i_col % n_f <= 7:
                five_group_idx[1].append(cnt)
            elif i_col % n_f <= 10:
                five_group_idx[2].append(cnt)
            elif i_col % n_f <= 14:
                five_group_idx[3].append(cnt)
            elif i_col % n_f <= 18:
                five_group_idx[4].append(cnt)
            if i_row % n_f <= 3:
                five_group_idx[0].append(cnt)
            elif i_row % n_f <= 7:
                five_group_idx[1].append(cnt)
            elif i_row % n_f <= 10:
                five_group_idx[2].append(cnt)
            elif i_row % n_f <= 14:
                five_group_idx[3].append(cnt)
            elif i_row % n_f <= 18:
                five_group_idx[4].append(cnt)
    five_group_idx[0] = np.array(five_group_idx[0])
    five_group_idx[1] = np.array(five_group_idx[1])
    five_group_idx[2] = np.array(five_group_idx[2])
    five_group_idx[3] = np.array(five_group_idx[3])
    five_group_idx[4] = np.array(five_group_idx[4])
    print('')
    return solution_space, five_group_idx

def inorder_walk(i,seq,n_tree_level):
    if i >= 2**(n_tree_level)-1:
        return []
    seq += inorder_walk(2*i+1,[],n_tree_level)
    seq += [i]
    seq += inorder_walk(2*i+2,[],n_tree_level)
    return seq