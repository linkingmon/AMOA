import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def index_to_matrix(idx, n_space=95, n_select=30):
    m = np.zeros((n_space,n_space))
    for i_idx in range(n_select):
        if idx[0,i_idx] >= idx[1,i_idx]:
            m[idx[0,i_idx]][idx[1,i_idx]] = 1
        else:
            m[idx[1,i_idx]][idx[0,i_idx]] = 1
    return m

n_select = 30
algor = ['SA', 'GA', 'PS', 'AC', 'DE']
idx_SA = np.load(open('model/idx_SA.npy', 'rb'))
idx_GA = np.load(open('model/idx_GA.npy', 'rb'))
idx_PS = np.load(open('model/idx_PS.npy', 'rb'))
idx_AC = np.load(open('model/idx_AC.npy', 'rb'))
idx_DE = np.load(open('model/idx_DE.npy', 'rb'))
# Random matrix

for i_algor in range(5):
    print(algor[i_algor])
    for i_class in range(5):
        print('Class %d' % i_class)
        print('$',end='')
        for i_select in range(n_select):
            if idx_SA[0][i_select][i_class] == idx_SA[1][i_select][i_class]:
                print('T_{',idx_SA[0][i_select][i_class],'}',end=',')
            else:
                print('T_{',idx_SA[0][i_select][i_class],',',idx_SA[1][i_select][i_class],'}',end=',')
        print('$')

# Define colormap
for i_class in range(5):
    n_space = 95
    data_1 = index_to_matrix(idx_SA[:,:,i_class])
    data_2 = index_to_matrix(idx_GA[:,:,i_class])
    data_3 = index_to_matrix(idx_PS[:,:,i_class])
    data_4 = index_to_matrix(idx_AC[:,:,i_class])
    data_5 = index_to_matrix(idx_DE[:,:,i_class])
    data = [data_1,data_2,data_3,data_4,data_5]
    data_total = data_1 + data_2 + data_3 + data_4 + data_5
    plt.clf()
    for i_algor in range(5):
        cmapmine = ListedColormap(['w', 'k'], N=2)
        plt.subplot(2,3,i_algor+1)
        plt.imshow(data[i_algor], cmap=cmapmine, vmin=0, vmax=1)
        plt.title(algor[i_algor])
        plt.xlabel('i')
        plt.xlabel('j')
    plt.subplot(2,3,6)
    plt.imshow(data_total, cmap='binary')
    plt.title('Total')
    plt.xlabel('i')
    plt.xlabel('j')
    plt.savefig('select/select_by_class_%d.jpg'% (i_class+1), dpi=300)

for i_algor in range(5):
    data_1 = eval('index_to_matrix(idx_' + algor[i_algor] + '[:,:,0])')
    data_2 = eval('index_to_matrix(idx_' + algor[i_algor] + '[:,:,1])')
    data_3 = eval('index_to_matrix(idx_' + algor[i_algor] + '[:,:,2])')
    data_4 = eval('index_to_matrix(idx_' + algor[i_algor] + '[:,:,3])')
    data_5 = eval('index_to_matrix(idx_' + algor[i_algor] + '[:,:,4])')
    data = [data_1,data_2,data_3,data_4,data_5]
    data_total = data_1 + data_2 + data_3 + data_4 + data_5
    for i_class in range(5):
        cmapmine = ListedColormap(['w', 'k'], N=2)
        plt.subplot(2,3,i_class+1)
        plt.imshow(data[i_class], cmap=cmapmine, vmin=0, vmax=1)
        plt.title('Class %d' % i_class)
        plt.xlabel('i')
        plt.xlabel('j')
    plt.subplot(2,3,6)
    plt.imshow(data_total, cmap='binary')
    plt.title('Total')
    plt.xlabel('i')
    plt.xlabel('j')
    plt.savefig('select/select_by_algorithm_%s.jpg'% algor[i_algor], dpi=300)