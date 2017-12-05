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
from scipy import ndimage
import scipy.linalg
import random
import matplotlib.pyplot as plt
import enchant

def principal_components(X,num):
    covx = np.cov(X, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - num, N - 1))
    v = np.fliplr(v)
    return v

def get_ten(X, model):
    labels = np.array(model['labels_train'])
    lst = sorted(set(labels))
    score = np.zeros(X.shape[1])
    for a, b in [(x,y) for x in range(len(lst)) for y in range(len(lst)) if x < y]:
    
        alst = X[labels[:] == lst[a],:]
        blst = X[labels[:] == lst[b],:]
        if (alst.shape[0] <= 1 or blst.shape[0] <= 1):
            continue
        score = np.add(score, divergence(alst, blst))
    sorted_score = np.argsort(-score)
    return sorted_score[0:10]

def divergence(class1, class2):
    """compute a vector of 1-D divergences
        
        class1 - data matrix for class 1, each row is a sample
        class2 - data matrix for class 2
        
        returns: d12 - a vector of 1-D divergence scores
        """
    
    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)
    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)
    
    return d12

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
    images = remove_noise(images_train)
    fvectors_train_full = images_to_feature_vectors(images, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    print('Reducing to 10 dimensions')
    v = principal_components(fvectors_train_full, 11)[:,1:11]
    model_data['v'] = v.tolist()
    model_data['mean'] = np.mean(fvectors_train_full).tolist()
    model_data['fvectors_train'] = np.dot((fvectors_train_full - np.mean(fvectors_train_full)), v).tolist()

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
    n = remove_noise(images_test)
    fvectors_test = images_to_feature_vectors(n, bbox_size)
    # Perform the dimensionality reduction.
    mean = np.array(model['mean'])
    v = np.array(model['v'])
    fvectors_test_reduced = np.dot((fvectors_test - mean), v)
    return fvectors_test_reduced

def remove_noise(page):
    n = []
    for l in page:
        image = ndimage.median_filter(l, 2)
        for i, row in enumerate(image):
            for j, px in enumerate(row):
                if(px <= 152):
                    image[i][j] = 0
                else:
                    image[i][j] = 255
        n.append(image)
#    plt.imshow(n[0],cmap='gray')
#    plt.show()
    return n

#def correct_errors(page, labels, bboxes, model):
#    """Dummy error correction. Returns labels unchanged.
#
#        parameters:
#
#        page - 2d array, each row is a feature vector to be classified
#        labels - the output classification label for each feature vector
#        bboxes - 2d array, each row gives the 4 bounding box coords of the character
#        model - dictionary, stores the output of the training stage
#        """
#    words = []
#    word = []
#    for i in range(len(bboxes) - 1):
#        word.append(labels[i])
##        print(abs(bboxes[i][2]-bboxes[i+1][0]), abs(bboxes[i][3] - bboxes[i+1][1]))
#        if(abs(bboxes[i][2]-bboxes[i+1][0]) >=7 or abs(bboxes[i][3] - bboxes[i+1][1]) >= 60):
#            words.append(word)
#            word = []
#    word.append(labels[len(labels) - 1])
#    words.append(word)
#
#    dictionary = enchant.Dict("en_GB")
#    new_words = []
#    l = [',',"'",'.','!']
#    for word in words:
#        w = ''.join(word)
#        end = []
#        if w[len(w) - 1] in l:
##            print(w, w[len(w) - 1])
#            end = w[len(w) - 1]
#            w = w[:len(w) - 1]
##            print(w)
#        last = ''
#        if(len(w) > 0 and dictionary.check(w) == False):
#            s = dictionary.suggest(w)
#            sugg = [x for x in s if len(x) == len(w)]
#            if(len(sugg) == 0):
##                print(w, sugg, w)
#                new_words.append(list(w))
#            else:
##                print(w, sugg, sugg[0])
#                new_words.append(list(sugg[0]))
#        else:
#            new_words.append(list(w))
#        if(len(end) > 0):
#            new_words.append(list(end))
#        print(new_words[len(new_words)-1])
#
#    letters = []
#    for word in new_words:
#        for char in word:
#            letters.append(char)
##    print(np.array(letters).shape)
##    print(labels.shape)
#    return np.array(letters)
def classify_page(page, model):
    """Dummy classifier. Always returns first label.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    print("classify")
    return classify(fvectors_train, labels_train, page, int(page.shape[0]/page.shape[1]))

def classify(train, train_labels, test, k, features=None):
    """Perform nearest neighbour classification."""
    
    # Use all feature is no feature parameter has been supplied
    if features is None:
        features=np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]
    
    nt = np.unique(train, axis=0)

#    nc = np.copy(train)
#    for k,x in enumerate(nc):
#        for i in range(1,2):
#            rand = random.randrange(5)
#            nx = np.copy(x)
#            nx = nx + rand
#            train = np.append(train, np.expand_dims(nx, axis=0), axis=0)
#            train_labels = np.append(train_labels, train_labels[k])


    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    x /= np.outer(modtest, modtrain.transpose()) # cosine distance
    dist = x
    nearest = np.argmax(dist, axis=1)

    kNearest = np.argsort(dist, axis=1)[:, ::-1][:,:k]
    labels = []
    w = [1/x for x in range(1,k+1)]
    for row in kNearest:
        a, b = np.unique(train_labels[row], return_index=True)
        t = np.array(list(zip(train_labels[row], w)))
        unique_weights = []
        for unique in a:
            ts = t[t[:,0] == unique, :]
            weights = ts[:, 1].astype(float)
            unique_weights.append(np.sum(weights, axis=0))
        i = np.argmax(np.array(unique_weights))
        labels.append(row[b[i]])
#        label = []
#        if(len(a) == 1):
#            label = a[0]
#        else:
#        totals = [0] * len(a)
#        for i, letter in enumerate(a):
#            for y in t:
#                if(y[0] == letter):
#                    totals[i] += y[1]
#        totals = np.array(totals)
#        label = a[np.argmax(totals)]

#        labels = np.append(labels, train_labels[0,:].tolist().index(label))
#    labels = labels.astype(int)
    return train_labels[labels]





