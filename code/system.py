"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
from scipy.stats import multivariate_normal


def principal_components(X,num):
    covx = np.cov(X, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - num, N - 1))
    v = np.fliplr(v)
    return v

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')

    v = principal_components(fvectors_train_full, 10)
    reduced = np.dot((fvectors_train_full - np.mean(fvectors_train_full)), v)
    model_data['v'] = v.tolist()
    model_data['mean'] = np.mean(fvectors_train_full).tolist()
    model_data['fvectors_train'] = reduced.tolist()

    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    mean = np.array(model['mean'])
    v = np.array(model['v'])
    fvectors_test_reduced = np.dot((fvectors_test - mean), v)
    return fvectors_test_reduced


def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    w, x, y, z = perce_fast(fvectors_train, labels_train, np.zeros((1, fvectors_train.shape[1] + 1)), 0.2)
    return z

def perce_fast(X, y, w_init, rho):
    """ perce_fast
        A more efficient implementation of the perceptron alogorithm
        For the notebook data this version will work x100 faster!
        
        X - the data matrix. Each row represents a separate sample
        y - a vector of integer class labels corresponding to the rows of X - labels must be +1 or -1
        w_init - the initial weight vector
        rho - a scalar learning rate
        """
    (N, nfeatures) = X.shape
    X = np.hstack((X, np.ones((N,1))))
    nfeatures += 1
    max_iter = 10000
    w = w_init
    iter = 0
    mis_class = N
    yy = np.tile(y, (1, nfeatures))
    while mis_class > 0 and iter < max_iter:
        iter += 1
        mc = ((np.dot(X, w.transpose()) * y) >= 0)[:, 0]
        z = np.dot(X, w.transpose()) * y
        mis_class = np.sum(mc)
        w -= rho * (np.sum(yy[mc, :] * X[mc, :], axis=0))
    return w, iter, mis_class, z
