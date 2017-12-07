"""Classification system.

name: Jake Sturgeon

version: v2.0
"""
import numpy as np
import utils.utils as utils
from scipy import ndimage
from scipy import spatial
import scipy.linalg
import random
import matplotlib.pyplot as plt
import enchant
import itertools
import warnings

def principal_components(X,n):
    """Compute the principal components of X to n dimensions
        source - lab 6
        
    Params:
    X - a matrix where each row is a vector
    n - used to reduce X to a n-D vector
    
    Returns:
    v - a reduced matrix
    """
    covx = np.cov(X, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - n, N - 1))
    v = np.fliplr(v)
    return v

def get_bounding_box_size(images):
    """Compute bounding box size given list of images.
    
    Params:
    images - a list of images stored as arrays
    
    Returns:
    height, width - of the images
    """
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
    
    Returns:
    fvectors - a matrix where each row is a vector
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

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.
        
    This function acts as the training stage. The images are loaded,
    noise is added to about a third of the data set and is then reduced
    to simualte the process of noise removal. These images are then
    turned into vectors and PCA is used to reduce the dimensions to 10.
    
    Params:
    train_page_names - list of training page names
    
    Returns:
    model_data - a dictionary that contains all the information
                 needed for the classification stage
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

    print("Simulating noise removal")
    images = process_noise(images_train)
    fvectors_train_full = images_to_feature_vectors(images, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions via PCA')
    v = principal_components(fvectors_train_full, 11)[:,1:11]
    model_data['v'] = v.tolist()
    model_data['mean'] = np.mean(fvectors_train_full).tolist()
    model_data['fvectors_train'] = np.dot((fvectors_train_full - np.mean(fvectors_train_full)), v).tolist()

    print("Training has finished")
    return model_data

def process_noise(images):
    random.seed(1) #1
    return remove_noise(add_noise(images, 3))

def add_noise(images, n):
    """ Add noise to some images.
        
    This loop is used to add varying levels of noise to about a 1/n of the training data set.
    This gives the training set knowledge for instances where noise is present
    
    Param:
    images - a list of images stored as arrays
    n - add varying levels of noise to 1/n of the image set
    
    Returns:
    images - The new set of images
    """
    
    k = 1
    for i, im in enumerate(images):
        if(i % n == 0):
            im = salt_pepper(im, k)
            k += 1
        if(k == n):
            k = 0
    return images

def salt_pepper(im, k):
    """ Add salt and pepper noise to image
        
    Param:
    image - an image
    k - probablity of the pixel being black, white, or no change
        
    Returns:
    image - The new image with added noise
    """
    
    ps = k / 10
    pp = k / 10
    # For each pixel, given a random value, change it to black or white
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            b = random.randint(0,99)/100
            if b < ps:
                im[i, j] = 0
            elif b > 1 - pp:
                im[i, j] = 1
    return im

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    
    Returns:
    fvectors_test_reduced - a 10-d feature vector with the vectors
    stored as rows of a matrix
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

def remove_noise(pages):
    """Load test data page.
        
    Params:
    pages - a list of pages
        
    Returns:
    pages - the list of pages after noise removal
    """
    warnings.filterwarnings('ignore')
    for l in pages:
        image = scipy.ndimage.filters.gaussian_filter(l, sigma=1.34)
        mean = image.mean()
        mean = np.nan_to_num(mean)
        for i, row in enumerate(image):
            for j, px in enumerate(row):
                if(px <= mean - (mean * 0.15)):
                    image[i][j] = 0
                else:
                    image[i][j] = 255
        l = image
    return pages

def classify_page(page, model):
    """Knn

    Params:
    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    
    Returns:
    The classified labels in a vector
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])
    return classify(fvectors_train, labels_train, page,int(page.shape[0]**(1/2)))

def classify(train, train_labels, test, k):
    """Knn
        
    Params:
    train - matrix, each row is training data
    train_labels - vector of the lables
    test - the data that is being classified
    k - check the k nearest neighbours
        
    Returns:
    The classified labels in a vector
    """
    
    dists = get_dists(train, test)
    
    """ Weighted Knn """
    kNearest = np.argsort(dists, axis=1)[:, ::-1][:,:k]
    labels = []
    w = [1/x for x in range(1,k+1)] # weights
    for row in kNearest:
        row_labels = train_labels[row]
        a, b = np.unique(row_labels, return_index = True)
        weighted_dists = np.array(list(zip(row_labels, w)))
        unique_weights = []
        for unique in a:
            letters = weighted_dists[weighted_dists[:,0] == unique, :]
            weights = letters[:, 1].astype(float)
            unique_weights.append(np.sum(weights, axis=0))
        i = np.argmax(np.array(unique_weights))
        labels.append(row[b[i]])

    return train_labels[labels]

def get_dists(train, test, features=None):
    """returns all the cosine distances
        
    Params:
    train - matrix, each row is training data
    test - the data that is being classified
        
    Returns:
    dists - matrix of distances
    """
    # source - lab 6
    # Use all feature is no feature parameter has been supplied
    if features is None:
        features=np.arange(0, train.shape[1])

    # Select the desired features from the training and test data
    train = train[:, features]
    test = test[:, features]

    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    x /= np.outer(modtest, modtrain.transpose()) # cosine distance
    dists = x # make coCal happy

    return dists

"""my attempt at divergence"""
#def get_ten(X, model):
#    labels = np.array(model['labels_train'])
#    lst = sorted(set(labels))
#    score = np.zeros(X.shape[1])
#    for a, b in [(x,y) for x in range(len(lst)) for y in range(len(lst)) if x < y]:
#
#        alst = X[labels[:] == lst[a],:]
#        blst = X[labels[:] == lst[b],:]
#        if (alst.shape[0] <= 1 or blst.shape[0] <= 1):
#            continue
#        score = np.add(score, divergence(alst, blst))
#    sorted_score = np.argsort(-score)
#    return sorted_score[0:10]

"""my attempt at error correction"""
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
#        if(abs(bboxes[i][2]-bboxes[i+1][0]) >=6 or abs(bboxes[i][3] - bboxes[i+1][1]) >= 60):
#            words.append(word)
#            word = []
#    word.append(labels[len(labels) - 1])
#    words.append(word)
#
#    dictionary = enchant.Dict("en_GB")
#    new_words = []
#    l = [',',"'",'.','!','’',';',':']
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
#
#            if(len(sugg) == 0):
##                print(w, sugg, w)
#                new_words.append(list(w))
#            else:
#                h = []
#                for s in sugg:
#                    count = sum(one != two for one, two in zip(s, w))
#                    h.append(count)
#                i = np.argmin(np.array(h))
#                new_words.append(list(sugg[i]))
#        else:
#            new_words.append(list(w))
#        if(len(end) > 0):
#            new_words.append(list(end))
#
#    letters = []
#    for word in new_words:
#        for char in word:
#            letters.append(char)
##    print(np.array(letters).shape)
##    print(labels.shape)
#    return np.array(letters)





""" source for code below http://norvig.com/spell-correct.html
    
    can be used for error correction but the word returned is sometimes less which loses information
"""
#import re
#from collections import Counter
#
#def words(text): return re.findall(r'\w+', text.lower())
#
#WORDS = Counter(words(open('big.txt').read()))
#
#def P(word, N=sum(WORDS.values())):
#    "Probability of `word`."
#    return WORDS[word] / N
#
#def correction(word):
#    "Most probable spelling correction for word."
#    return max(candidates(word), key=P)
#
#def candidates(word):
#    "Generate possible spelling corrections for word."
#    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
#
#def known(words):
#    "The subset of `words` that appear in the dictionary of WORDS."
#    return set(w for w in words if w in WORDS)
#
#def edits1(word):
#    "All edits that are one edit away from `word`."
#    letters    = 'abcdefghijklmnopqrstuvwxyz'
#    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#    deletes    = [L + R[1:]               for L, R in splits if R]
#    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#    inserts    = [L + c + R               for L, R in splits for c in letters]
#    return set(deletes + transposes + replaces + inserts)
#
#def edits2(word):
#    "All edits that are two edits away from `word`."
#    return (e2 for e1 in edits1(word) for e2 in edits1(e1))




