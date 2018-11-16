import pandas as pd
import numpy as np
import platform
import sys
#input_filename = sys.argv[1]
#output_filename = sys.argv[2]
input_filename = 'test_X'
output_filename = 'ans.csv'
test_X = pd.read_csv(input_filename).values.astype(np.float64)
test_X = np.hstack((np.ones((test_X.shape[0],1)),test_X))
w = np.load('w_generative.npy')
def sigmoid(z):
    return ( np.sign((1-np.sign(z))) + np.sign(z)*1./(1.+np.exp(-abs(z)))    )
temp = (sigmoid(test_X.dot(w)).T)[0]
content = 'id,label\n'
for i,value in enumerate(temp):
    if value>0.5:
        content += '{0},0\n'.format(i+1)
    else:
        content += '{0},1\n'.format(i+1)
with open(output_filename,'w') as f:
    f.write(content)
