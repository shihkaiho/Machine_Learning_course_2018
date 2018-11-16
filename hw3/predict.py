import pandas as pd
import numpy as np
from keras.models import load_model
import sys
input_filename = sys.argv[1]
output_filename = sys.argv[2]
#input_filename = 'data_set\\test.csv'
#output_filename = 'ans.csv'
RawData = pd.read_csv(input_filename)
data_num = RawData.shape[0]
dim = np.fromstring(RawData['feature'].values[0],\
                    dtype=np.float64, sep=' ').shape[0]
dim2D = int(np.sqrt(dim))
test_X = np.empty((data_num, dim2D, dim2D, 1))
for i in range(data_num):
    test_X[i,:,:,0] = np.fromstring(RawData['feature'].values[i],\
                                   dtype=np.float64,\
                                   sep=' ').reshape(dim2D,dim2D)/255.
model = load_model('CNN.h5')
result = model.predict(test_X)
result = result.argmax(axis=1)
id = np.arange(data_num)
output = pd.DataFrame({'id': id,'label':result})
output.to_csv(output_filename, index = False) 
