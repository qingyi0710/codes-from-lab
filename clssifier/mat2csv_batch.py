import pandas as pd
import scipy
import os
from scipy import io
from sklearn.decomposition import PCA


# features_struct = scipy.io.loadmat('../data/ddsmfea.mat')
num = 0.0
path = '../CCA+PCA/cca_result'+str(num)+'/'
save_dir = '../data/ddsm/cca_feature_test/cca_feature_test'+str(num)+'/'


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
# print(features_struct['ddsmfea'])
"""
saving features to csv
"""
for name in os.listdir(path):
    print(name)
    features_struct = scipy.io.loadmat(path + name)
    feature5 = features_struct[name.split('.')[0]][:]
    print(feature5.shape)
    dfdata_feature = pd.DataFrame(feature5)
    datapath_feature = save_dir + name.split(".")[0]+'.csv'
    dfdata_feature.to_csv(datapath_feature,index=False,header=False)


# path1 = '../data/ddsm/cca_mat_test/'
#
# for folder in os.listdir(path1):
#     path = path1+folder+'/'
#     for name in os.listdir(path):
#
#         features_struct = scipy.io.loadmat(path + name)
#         feature5 = features_struct[name.split('.')[0]][:]
#         print(feature5.shape)
#         save_dir = '../data/ddsm/cca_feature_test'+folder[-3:]+'/'
#         if not os.path.isdir(save_dir):
#             os.makedirs(save_dir)
#         dfdata_feature = pd.DataFrame(feature5)
#         datapath_feature = save_dir + name.split(".")[0] + '.csv'
#         dfdata_feature.to_csv(datapath_feature, index=False, header=False)





# for name in os.listdir(path):
#     features_struct = scipy.io.loadmat(path+name)
#     feature5 = features_struct['ddsmfea']
#     dfdata_feature = pd.DataFrame(feature5)
#     pca = PCA(50)
#     feature_map = pca.fit_transform(dfdata_feature)
#     print(feature_map.shape)
#     datapath_feature = save_dir + '50' + name.split('.')[0]
#     feature_data = {'EMK': feature_map}
#     sio.savemat(datapath_feature, feature_data )


"""
saving labels to csv
"""
# feature6 = features_struct['material'][:]
# print(feature6.shape)
# dfdata_label = pd.DataFrame(feature6)
# datapath_label = save_dir + 'sift_labels_mixing0.5_73.csv'
# dfdata_label.to_csv(datapath_label, index=False, header=False)
#
