import pandas as pd
import numpy as np
import platform
import os
import sys
RawData_X = pd.read_csv(sys.argv[0]).values
RawData_Y = pd.read_csv(sys.argv[1]).values
RawDataMax = np.max(RawData_X,axis=0)
RawData_X = RawData_X/RawDataMax
train_X = RawData_X
train_Y = RawData_Y
train_X0 = RawData_X[np.where(RawData_Y==0)[0]].astype(np.float64)
N0 = train_X0.shape[0]

train_X1 = RawData_X[np.where(RawData_Y==1)[0]].astype(np.float64)
N1 = train_X1.shape[0]

mu0 = train_X0.mean(axis=0).reshape(-1,1)
mu1 = train_X1.mean(axis=0).reshape(-1,1)

sigma0 = np.cov(train_X0,rowvar=False)
sigma1 = np.cov(train_X1,rowvar=False)
sigma = (sigma0*N0 + sigma1*N1)/(N0+N1)
sigma_inv = np.linalg.pinv(sigma)

w = sigma_inv.dot(mu1 - mu0)
b = -0.5*(mu1.T).dot(sigma_inv).dot(mu1) + 0.5*(mu0.T).dot(sigma_inv).dot(mu0) + np.log(N1/N0)
w = np.vstack((b,w))

dim = train_X.shape[1] + 1
data_num = train_X.shape[0]
train_X = np.hstack((np.ones((data_num,1)),train_X))
zeros = np.zeros((data_num,1))

def sigmoid(z):
    return ( np.sign((1-np.sign(z))) + np.sign(z)*1./(1.+np.exp(-abs(z)))    )
def Loss_fun():
    z = train_X.dot(w)
    return (np.fmax(z, zeros) - z * train_Y + np.log(1. + np.exp(-abs(z)))).mean()
def accuracy():
    temp = sigmoid(train_X.dot(w))
    temp_2 = train_Y[np.where(temp>0.5)[0]]
    P0_correct = np.where(temp_2==1)[0].shape[0]
    temp_2 = train_Y[np.where(temp<=0.5)[0]]
    P1_correct = np.where(temp_2==0)[0].shape[0]
    C=(P0_correct+P1_correct)/data_num
    return C
#def Loss_fun():
#    return (train_Y * -np.log(sigmoid(train_X.dot(w)+b)) + (1 - train_Y) * -np.log(1 - sigmoid(train_X.dot(w)+b))).sum()
# In[ ]:
iteration = 0
#w = np.zeros((dim,1))
print_freq = 1000
LearningRate = np.ones((dim,1))*6e-4
sigma2 = np.zeros((dim,1))
grad = np.zeros((dim,1))
Loss_last = 0.
threshold = 1e-5
alpha = 0.98
while 1:
    if iteration%print_freq == 0:
        Loss_curr = Loss_fun()
        print('{0}\t {1:.5f}\t {2:.5f}'.format(iteration,Loss_curr,accuracy()))
        if abs(Loss_curr - Loss_last) <= threshold:
            break
        else:
            Loss_last = Loss_curr
    grad = (train_X.T).dot( train_Y-sigmoid(train_X.dot(w)) )
    sigma2 = alpha * sigma2 + (1-alpha) * grad**2
    #sigma2 += grad**2
    w += LearningRate/np.sqrt(sigma2)*grad
    iteration += 1
print('=== end ===')

# In[ ]:
C= accuracy()
print(C)

# %%
if platform.system() == 'Windows':
    test_X = pd.read_csv('..\\data_set\\test_X').values.astype(np.float64)
else:
    test_X = pd.read_csv('../data_set/test_X').values.astype(np.float64)
test_X = test_X/RawDataMax
test_X = np.hstack((np.ones((test_X.shape[0],1)),test_X))
temp = (sigmoid(test_X.dot(w)).T)[0]
content = 'id,label\n'
for i,value in enumerate(temp):
    if value>0.5:
        content += '{0},1\n'.format(i+1)
    else:
        content += '{0},0\n'.format(i+1)
if platform.system() == 'Windows':
    DirName0 = '..\\commits\\logistic_normalize\\'
else:
    DirName0 = '..\\commits\\logistic_normalize\\'
if not os.path.exists(DirName0):
    os.mkdir(DirName0)
if platform.system() == 'Windows':
    f = open(DirName0+'L={0:.5f}_C={1:.5f}.csv'.format(Loss_curr,C), 'w')
else:
    f = open(DirName0+'L={0:.5f}_C={1:.5f}.csv'.format(Loss_curr,C), 'w')
f.write(content)
f.close()
if platform.system() == 'Windows':
    DirName1 = '..\\models\\logistic\\n_L={0:.5f}_C={1:.5f}\\'.format(Loss_curr,C)
else:
    DirName1 = '../models/logistic/n_L={0:.5f}_C={1:.5f}/'.format(Loss_curr,C)
if not os.path.exists(DirName1):
    os.mkdir(DirName1)
if platform.system() == 'Windows':
    np.save(DirName1+'w.npy',w)
else:
    np.save(DirName1+'w.npy',w)

