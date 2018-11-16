import pandas as pd
from os.path import join as path
from keras.models import load_model
import sys

#test_path = path('..','data','test.csv')
test_path = sys.argv[1]
#model_path = path('..','temp','model_best.h5')
model_path = 'model_best.h5'
#output_path = path('..','temp','test.csv')
output_path = sys.argv[2]

test_data = pd.read_csv(test_path).values[:,1:]
model = load_model(model_path)
predict =  model.predict([test_data[:,0], test_data[:,1]],verbose=1)
with open(output_path,'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(predict.shape[0]):
        f.write('{0},{1}\n'.format(i+1,predict[i][0]))