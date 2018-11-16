import numpy as np
import pandas as pd
import sys
image_path = sys.argv[1]
test_case_path = sys.argv[2]
#test_case_path = 'test_case.csv'
prediction_path = sys.argv[3]
#prediction_path = 'test_ans.csv'
labels = np.load('labels.npy')
test_case = pd.read_csv(test_case_path).values[:,1:]
comp = np.logical_not(np.logical_xor(labels[test_case[:,0]],labels[test_case[:,1]]))
predict = np.hstack((np.arange(comp.shape[0]).reshape(-1,1), comp.reshape(-1,1)))
predict_df = pd.DataFrame(data = predict, columns = ['ID', 'Ans'])
predict_df.to_csv(prediction_path, index = False)
