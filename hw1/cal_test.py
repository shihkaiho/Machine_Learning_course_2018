import numpy as np
import pandas as pd
import sys
assert(len(sys.argv)==3)
input_filename = sys.argv[1]
output_filename = sys.argv[2]
w = np.load('model/w.npy')
Xmean = np.load('model/Xmean.npy')
Xstd = np.load('model/Xstd.npy')
Ystd = np.load('model/Ystd.npy')
Ymean = np.load('model/Ymean.npy')
Corr_filter_idx = np.load('model/Corr_filter_idx.npy')
RawData = pd.read_csv(input_filename, index_col=0, header=None).values
RawData[RawData == 'NR'] = 0.
RawData = RawData[:,1:].astype(np.float64)
test_num = int(RawData.shape[0]/18)
content = 'id,value\n'
temp = np.ones((1,Corr_filter_idx[0].size+1))
for i in range(test_num):
    temp[0,1:] = RawData[18*i:18*i+18, :][Corr_filter_idx[0],Corr_filter_idx[1]]
    temp = (temp - Xmean)/Xstd
    ans = temp.dot(w)
    ans = temp.dot(w)*Ystd + Ymean
    if ans[0,0]<0:
        ans[0,0] = 0
    content += 'id_{0},{1}\n'.format(i,ans[0,0])
f = open(output_filename, 'w')
f.write(content)
f.close()
