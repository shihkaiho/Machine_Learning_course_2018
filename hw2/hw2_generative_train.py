import pandas as pd
import numpy as np
import platform
import sys
RawData_X = pd.read_csv(sys.argv[1]).values
RawData_Y = pd.read_csv(sys.argv[2]).values
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

w = sigma_inv.dot(mu0 - mu1)
b = -0.5*(mu0.T).dot(sigma_inv).dot(mu0) + 0.5*(mu1.T).dot(sigma_inv).dot(mu1) + np.log(N0/N1)
w = np.vstack((b,w))
train_X0 = np.hstack((np.ones((train_X0.shape[0],1)),train_X0))
train_X1 = np.hstack((np.ones((train_X1.shape[0],1)),train_X1))

# sigmoid(wx+b)
# w: row vector
# x: column vectors
# b: constant
def sigmoid(z):
    return ( np.sign((1-np.sign(z))) + np.sign(z)*1./(1.+np.exp(-abs(z)))    )
 

temp = sigmoid(train_X0.dot(w))
P0_correct = np.where(temp>0.5)[0].shape[0]
temp = sigmoid(train_X1.dot(w))
P1_correct = np.where(temp<=0.5)[0].shape[0]

print((P0_correct+P1_correct)/RawData_X.shape[0])

# %%
if platform.system() == 'Windows':
    test_X = pd.read_csv('..\\data_set\\test_X').values.astype(np.float64)
else:
    test_X = pd.read_csv('../data_set/test_X').values.astype(np.float64)
test_X = np.hstack((np.ones((test_X.shape[0],1)),test_X))
temp = sigmoid(test_X.dot(w))[0]
content = 'id,label\n'
for i,value in enumerate(temp):
    if value>0.5:
        content += '{0},0\n'.format(i+1)
    else:
        content += '{0},1\n'.format(i+1)
if platform.system() == 'Windows':
    f = open('..\\commits\\generative.csv', 'w')
else:
    f = open('../commits/generative.csv', 'w')
f.write(content)
f.close()
if platform.system() == 'Windows':
    np.save('..\\models\\generative\\w.npy',w)
else:
    np.save('../models/generative/w.npy',w)