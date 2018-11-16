from skimage import io
import numpy as np
from os.path import join as path
from os import listdir
import sys
#dirname = path('data_set','PCA')
dirname = sys.argv[1]
#recon_name = '414.jpg'
recon_name = sys.argv[2]
def reconstruct_img(dirname,recon_name,num=4):
    pictures_filenames = listdir(dirname)
    data_num = len(pictures_filenames)
    dim = io.imread(path(dirname,pictures_filenames[0])).flatten().shape[0]
    data = np.empty((data_num,dim),dtype=np.float64)
    for idx, pic_name in enumerate(pictures_filenames):
        img = io.imread(path(dirname,pic_name)).flatten()
        data[idx,:] = img
    X_mean = data.mean(axis=0)
    data -= X_mean
    data = data.T
    eigenfaces, s, vh = np.linalg.svd(data, full_matrices=False)
    eigenfaces = eigenfaces.T
    eigenfaces = eigenfaces[:num,:]
    recon_idx = pictures_filenames.index(recon_name)
    recon_img = data[:,recon_idx]
    weights = (recon_img*eigenfaces).sum(axis=1)
    reconed_img = (weights.reshape(-1,1)*eigenfaces).sum(axis=0)
    reconed_img += X_mean
    reconed_img -= reconed_img.min()
    reconed_img /= reconed_img.max()
    reconed_img = (reconed_img * 255).astype(np.uint8).reshape((600,600,3))
    io.imsave('reconstruction.png',reconed_img)
    return
#%%
if __name__ == '__main__':
    reconstruct_img(dirname,recon_name,num=4)





