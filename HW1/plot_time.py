import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt

with open("time/SA.list", "rb") as fp:
    time_SA = np.array(pickle.load(fp))
with open("time/GA.list", "rb") as fp:
    time_GA = np.array(pickle.load(fp))
with open("time/PS.list", "rb") as fp:
    time_PS = np.array(pickle.load(fp))
with open("time/AC.list", "rb") as fp:
    time_AC = np.array(pickle.load(fp))
with open("time/DE.list", "rb") as fp:
    time_DE = np.array(pickle.load(fp))

time_SA_mean = np.mean(time_SA,axis=0)
time_GA_mean = np.mean(time_GA,axis=0)
time_PS_mean = np.mean(time_PS,axis=0)
time_AC_mean = np.mean(time_AC,axis=0)
time_DE_mean = np.mean(time_DE,axis=0)

for i_class in range(5):
    plt.clf()
    plt.plot(time_SA[i_class,:,0],time_SA[i_class,:,1])
    plt.plot(time_GA[i_class,:,0],time_GA[i_class,:,1])
    plt.plot(time_PS[i_class,:,0],time_PS[i_class,:,1])
    plt.plot(time_AC[i_class,:,0],time_AC[i_class,:,1])
    plt.plot(time_DE[i_class,:,0],time_DE[i_class,:,1])
    plt.legend(['SA','GA','PS','AC','DE'])
    plt.xlabel('Time (s)')
    plt.ylabel('Accuracy')
    plt.title('Acc-Time of Class %d' % (i_class+1))
    plt.savefig('time/analysis%d.jpg' % (i_class))


plt.clf()
plt.plot(time_SA_mean[:,0],time_SA_mean[:,1])
plt.plot(time_GA_mean[:,0],time_GA_mean[:,1])
plt.plot(time_PS_mean[:,0],time_PS_mean[:,1])
plt.plot(time_AC_mean[:,0],time_AC_mean[:,1])
plt.plot(time_DE_mean[:,0],time_DE_mean[:,1])
plt.legend(['SA','GA','PS','AC','DE'])
plt.xlabel('Time (s)')
plt.ylabel('Accuracy')
plt.title('Acc-Time')
plt.savefig('time/analysis_mean.jpg')