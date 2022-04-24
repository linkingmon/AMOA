import numpy as np

def get_score(train_y, predict_y, n_y=5):
    # print(train_y.shape, predict_y.shape)
    acc = []
    for i_class in range(n_y):
        acc.append(np.sum(train_y[:,i_class] == predict_y[:,i_class]) / train_y.shape[0])
    return acc, np.mean(acc)