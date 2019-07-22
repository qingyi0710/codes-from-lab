import pandas as pd
import scipy
import os
from scipy import io
import h5py
from sklearn.decomposition import PCA
import scipy.io as sio



# features_struct = scipy.io.loadmat('../data/ddsmfea.mat')
path = '../data/ddsm/feature_mat/'
save_dir = '../data/ddsm/feature/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
# print(features_struct['ddsmfea'])
name = 'gist.mat'

# features_struct = scipy.io.loadmat(path+name)
features_struct = h5py.File(path+name,'r')
feature5 = features_struct['GistFeats']
dfdata_feature = pd.DataFrame(list(feature5))
datapath_feature = save_dir + name.split('.')[0]+'.csv'
dfdata_feature.to_csv(datapath_feature, index=False, header=False)


"""
saving features to csv
"""

# features_struct = scipy.io.loadmat(path+'vgg.mat')
# feature5 = features_struct['vgg16']
# dfdata_feature = pd.DataFrame(feature5)
# print(dfdata_feature.shape)
# datapath_feature = save_dir + 'vgg.csv'
# dfdata_feature.to_csv(datapath_feature, index=False, header=False)

"""
saving labels to csv
"""
# feature6 = features_struct['material'][:]
# print(feature6.shape)
# dfdata_label = pd.DataFrame(feature6)
# datapath_label = save_dir + 'sift_labels_mixing0.5_73.csv'
# dfdata_label.to_csv(datapath_label, index=False, header=False)
#
