from hog_feature import get_img_datas
import os, numpy as np, pandas as pd
import scipy.io as sio

#save_type_list = [0.5, 1, 1.5, 2]
# save_type_rank = 2

# save_dir = "../hog_classic_result/"
# filepath = '../classic_add'+str(save_type_rank)+'/'

save_dir = "G:/no_augment/result/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = 'G:/tumor_dataset/'

n_dimension = 500

feature_map, name_list, label_list = get_img_datas(filepath, n_dimension)

"""
saving features to csv
"""


# savepath_feature= save_dir+'hog_features_mixing73.csv'
# feature_data = pd.DataFrame(feature_map)
# feature_data.to_csv(savepath_feature, index=False, header=False)
# savepath_feature = save_dir+'hog_features.mat'
# feature_data = {'hog': feature_map}
# sio.savemat(savepath_feature, feature_data)


"""
saving labels to csv
"""

# savepath_label= save_dir+'hog_labels_mixing73.csv'
# lavel_data = pd.DataFrame(label_list)
# lavel_data.to_csv(savepath_label, index=False, header=False)
# savepath_label= save_dir+'hog_labels.mat'
# lavel_data = {'material': label_list}
# sio.savemat(savepath_label, lavel_data)
