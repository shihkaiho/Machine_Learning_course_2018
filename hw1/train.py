import numpy as np
import pandas as pd
import os
import platform
import time
'''
Read Data
'''
tStart = time.time()
if platform.system() == 'Windows':
    RawData = pd.read_csv('train.csv', encoding='big5').values
else:
    RawData = pd.read_csv('train.csv', encoding='big5').values
RawData = RawData[:,3:] #slice data
RawData[RawData == 'NR'] = 0
data = np.zeros((12,18,480)) # mounth, species, value
for month in range(12): #per month
    for species in range(18): # per species
        for date in range(20): #first 20 days in a month
            data[month, species, 24*date:24*(date+1)]\
            = RawData[18*20*month+18*date + species, 0:]
del RawData
print('read data time elapsed: {0:.3f}s'.format(time.time()-tStart))
'''
process data for correlation and fiter data
'''
subset_num = 471
month_num = 12
dim = 18*9
task_num = subset_num * month_num
# Y = Xw
Y = np.empty((task_num,1))
X = np.ones((task_num,dim))
task_num = 0
filter_data = np.array([0,1,2,3,7,8,9,11,12,13,14,15,16,17])
# construct X matrix
for month in range(month_num):
    for subset in range(subset_num):
        temp = data[month,:,subset:subset+9][filter_data]
        tempPM = data[month,9,subset:subset+9]
        Y[task_num] = data[month,9,subset+9]
        if (temp[temp<=0].size>=10) or (Y[task_num] <= 0) or\
           (tempPM[tempPM>600].size>0) or (Y[task_num] > 600) or\
           (tempPM[tempPM<=0].size>0):
            continue
        X[task_num,:] = data[month,:,subset:subset+9].reshape(1,-1)
        task_num += 1
X = X[:task_num,:]
Y = Y[:task_num,:]
Xnor = X - X.mean(axis=0)
Ynor = Y - Y.mean()
# Sxy/SxxSyy
Correlation = (Ynor.T).dot(Xnor)/np.sqrt((Xnor**2).sum(axis = 0))/np.sqrt((Ynor**2).sum())
Correlation.resize(18,9)
valve = -10
Corr_filter_idx = np.where( Correlation>valve )
'''
process data for regression and filter data
'''
subset_num = 471
month_num = 12
dim = Corr_filter_idx[0].size + 1
task_num = month_num*subset_num
# Y = Xw
Y = np.empty((task_num,1))
X = np.ones((task_num,dim))
task_num = 0
# construct X matrix
for month in range(month_num):
    for subset in range(subset_num):
        tempALL = data[month,:,subset:subset+9][filter_data]
        tempPM = data[month,9,subset:subset+9]
        Y[task_num] = data[month,9,subset+9]
        if (tempALL[tempALL<=0].size>=10) or (Y[task_num] < 0) or\
           (tempPM[tempPM>600].size>0) or (Y[task_num] > 600) or\
           (tempPM[tempPM<0].size>0):
            continue
        X[task_num,1:] = data[month,:,subset:subset+9][Corr_filter_idx]
        task_num += 1
X = X[:task_num,:]
Y = Y[:task_num]
del data
'''
normalize
'''
Xmean = X.mean(axis=0)
Xmean[0] = 0
Xstd = X.std(axis=0)
Xstd[0] = 1
Ymean = Y.mean()
Ystd = Y.std()
X = (X - Xmean)/Xstd
Yorigin = Y
Y = (Y - Ymean)/Ystd
def Loss_fun(w):
    return np.sqrt(((Yorigin - X.dot(w)*Ystd - Ymean)**2).mean())
def Loss_analytic():
    return (np.linalg.inv((X.T).dot(X)).dot(X.T)).dot(Y)
#def Loss_fun_split(w):
#    return np.sqrt(((trainYorigin - trainX.dot(w)*Ystd - Ymean)**2).mean())
#def Loss_analytic_split():
#    return (np.linalg.inv((trainX.T).dot(trainX)).dot(trainX.T)).dot(trainY)
'''
cross validation
'''
split = 5
seed = np.arange(X.shape[0])
np.random.shuffle(seed)
X = X[seed]
Y = Y[seed]
Yorigin = Yorigin[seed]
X_split = np.array_split(X, split)
Y_split = np.array_split(Y, split)
Yorigin_split = np.array_split(Yorigin, split)
print_t = 5000
print_output = ""
lamda = 0.
threshold = 1e-5
start_ada = 6.4
'''
for i in range(split):
    testX = X_split[i]
    trainX = np.vstack(np.delete(X_split,i,axis=0))
    testY = Y_split[i]
    trainY = np.vstack(np.delete(Y_split,i,axis=0))
    testYorigin = Yorigin_split[i]
    trainYorigin = np.vstack(np.delete(Yorigin_split,i,axis=0))
    #iteration
    tStart = time.time()
    iteration = 0
    Loss_curr = 0
    LearningRate = np.ones((dim,1))*1e-3
    sigma2 = np.zeros((dim,1))
    grad = np.zeros((dim,1))
    #w = np.zeros((dim,1))
    w = Loss_analytic_split()
    while 1:
        if iteration%print_t == 0:
            temp = Loss_fun_split(w)
            print('{0}\t {1:.5f}'.format(iteration,temp))
            if abs(temp - Loss_curr) <= threshold:
                break
            else:
                Loss_curr = temp
            if temp > start_ada:
                sigma2 = np.zeros((dim,1))
        grad = (trainX.T).dot((trainY - trainX.dot(w))) - lamda * w
        sigma2 += grad**2
        w += LearningRate/np.sqrt(sigma2)*grad
        iteration += 1
    #Validation
    test_error = np.sqrt(((testYorigin - testX.dot(w)*Ystd - Ymean)**2).mean())
    print_output += 'set {0}\t train: {1:.5f} test {2:.5f}\n'.format(i, Loss_fun_split(w), test_error)
    print('set {0}\t train: {1:.5f} test {2:.5f}'.format(i, Loss_fun_split(w), test_error))

'''
'''
iteration
'''
iteration = 0
LearningRate = np.ones((dim,1))*1e-3
sigma2 = np.zeros((dim,1))
grad = np.zeros((dim,1))
Loss_curr = 0
w = np.zeros((dim,1))
#w = Loss_analytic()
while 1:
    if iteration%print_t == 0:
        temp = Loss_fun(w)
        print('{0}\t {1:.5f}'.format(iteration,temp))
        if abs(temp - Loss_curr) <= threshold:
            break
        else:
            Loss_curr = temp
        if temp > start_ada:
            sigma2 = np.zeros((dim,1))
    grad = X.T.dot((Y - X.dot(w))) - lamda * w
    sigma2 += grad**2
    w += LearningRate/np.sqrt(sigma2)*grad
    iteration += 1
print_output += 'set all\t train: {0:.5f}\n'.format(Loss_fun(w))
print(print_output)
'''
Calculate test data error and Save data
'''
if platform.system() == 'Windows':
    RawData = pd.read_csv('test.csv',index_col = 0, encoding='big5', header=None).values
else:
    RawData = pd.read_csv('test.csv',index_col = 0, encoding='big5', header=None).values
RawData[RawData == 'NR'] = 0.
RawData = RawData[:,1:].astype(np.float64)
test_num = int(RawData.shape[0]/18)
content = 'id,value\n'
temp = np.ones((1,dim))
for i in range(test_num):
    temp[0,1:] = RawData[18*i:18*i+18,:][Corr_filter_idx]
    temp = (temp - Xmean)/Xstd
    ans = temp.dot(w)*Ystd + Ymean
    if ans[0,0]<=0:
        ans[0,0] = 0
    content += 'id_{0},{1}\n'.format(i,ans[0,0])
error = Loss_fun(w)
# make dir
if platform.system() == 'Windows':
    DirName1 = 'model\\'
else:
    DirName1 = 'model/'
if not os.path.exists(DirName1):
    os.mkdir(DirName1)
np.save(DirName1+'w.npy',w)
np.save(DirName1+'Xmean.npy',Xmean)
np.save(DirName1+'Xstd.npy',Xstd)
np.save(DirName1+'Ymean.npy',Ymean)
np.save(DirName1+'Ystd.npy',Ystd)
np.save(DirName1+'Corr_filter_idx.npy',Corr_filter_idx)
f = open(DirName1+'ans.csv', 'w')
f.write(content)
f.close()