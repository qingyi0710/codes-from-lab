import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter
import matplotlib.image as mpimg  # mpimg 用于读取图片
import os
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def extract_features(imgs, feature_fns, verbose=False):
    """
  Given pixel data for images and several feature functions that can operate on
  single images, apply all feature functions to all images, concatenating the
  feature vectors for each image and storing the features for all images in
  a single matrix.

  Inputs:
  - imgs: N x H X W X C array of pixel data for N images.
  - feature_fns: List of k feature functions. The ith feature function should
    take as input an H x W x D array and return a (one-dimensional) array of
    length F_i.
  - verbose: Boolean; if true, print progress.

  Returns:
  An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
  of all features for a single image.
  """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print('Done extracting features for %d / %d images' % (i, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

    Parameters:
      rgb : RGB image

    Returns:
      gray : grayscale image
  
  """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image
  
       Modified from skimage.feature.hog
       http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog
     
     Reference:
       Histograms of Oriented Gradients for Human Detection
       Navneet Dalal and Bill Triggs, CVPR 2005
     
    Parameters:
      im : an input grayscale or rgb image
      
    Returns:
      feat: Histogram of Gradient (HOG) feature
    
  """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.atleast_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx // 2::cx, cy // 2::cy]
        # use // to avoid the integer problem
        # orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx // 2::cx, cy // 2::cy].T
    # print(orientation_histogram)

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
  Compute color histogram for an image using hue.

  Inputs:
  - im: H x W x C array of pixel data for an RGB image.
  - nbin: Number of histogram bins. (default: 10)
  - xmin: Minimum pixel value (default: 0)
  - xmax: Maximum pixel value (default: 255)
  - normalized: Whether to normalize the histogram (default: True)

  Returns:
    1D vector of length nbin giving the color histogram over the hue of the
    input image.
  """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist


def get_img_datas(filepath, n_dimension):
    """
    :param filepath:
    :param n_dimension: must be smaller than the number of image in a folder in total
    :return:
    """
    # folder = os.listdir(filepath)

    folder = ['neg', 'pos']
    print(folder)
    # print(folder)
    name_list = []
    label_list = []
    feature_map = []
    i = 0
    for name in folder:
        i += 1
        folder_num = len(os.listdir(filepath + name))
        count = 0
        for img_name in os.listdir(filepath + name):
            count += 1
            print('class : [%d / %d] image: [ %d / %d] folder: %s   image name : %s' % (
            i, len(folder), count, folder_num, name, img_name))
            im = mpimg.imread(filepath + name + '/' + img_name)
            feature = hog_feature(im)  # extract feature
            feature_map.append(feature)
            name_list.append(img_name)
            label_list.append(name)

    #  decrease the dimension of features
    pca = PCA(n_dimension)
    feature_map = pca.fit_transform(feature_map)
    return feature_map, name_list, label_list

# def extract_hog_feature(filepath):
#     # print(len(os.listdir(r"../benign/")))
#     feature_map = []
#     for file in os.listdir(filepath):  # Read the file directory of the previous level
#         print(file)
#         # file_d = os.path.join(filepath, file)
#         im = mpimg.imread(filepath+file)
#         feature = hog_feature(im)  # extract  feature
#         feature_map.append(feature)  # Sequentially assign
#     feature_map = np.array(feature_map)
#     return feature_map
