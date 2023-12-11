"""
    Dynamic Time Warping toolset.

    Based on DTW from http://alexminnaar.com/time-series-classification-and-clustering-with-python.html

    Also see here: 
    https://github.com/alexminnaar/time-series-classification-and-clustering
"""

import random

import numpy as np
import pandas as pd
from scipy import spatial
from numba import njit


@njit
def dtw_distance(s1, s2):
    """                                              
    Dynamic time warping finds the optimal non-linear alignment between two time series s1 and s2.
    
    :param s1: first series 
    :param s2: secod series 
    :return: distance 
    """
    dtw = {}

    for i in range(len(s1)):
        dtw[(i, -1)] = np.inf

    for i in range(len(s2)):
        dtw[(-1, i)] = np.inf

    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])


@njit
def dtw_window_distance(s1, s2, w):
    """
    Speed up version of dynamic time warping. This works under the assumption that it is unlikely for q_i and c_j 
    to be matched if i and j are too far apart. The threshold is determined by a window size w. 
    
    :param s1: first series 
    :param s2: secod series 
    :param w: window size 
    :return: distance 
    """
    dtw = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            dtw[(i, j)] = np.inf

    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])

    return np.sqrt(dtw[len(s1) - 1, len(s2) - 1])


def dtw_keogh_lower_bound(s1, s2, r):
    """
    Lower bound Keogh method of dynamic time warping
    
    :param s1: 
    :param s2: 
    :param r: 
    :return: 
    """
    lb_sum = 0

    for ind, i in enumerate(s1):
        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            lb_sum = lb_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            lb_sum = lb_sum + (i - lower_bound) ** 2

    return np.sqrt(lb_sum)


def knn(x_train, y_train, x_test, w):
    """
    The following is the 1-NN algorithm that uses dynamic time warping Euclidean distance. 
    Here x_train is the training set of time series examples where the class that the time series belongs
    to appropriate classes from y_train time series.
    x_test is the test set whose corresponding classes you are trying to predict.

    In this algorithm, for every time series in the test set, a search must be performed through all points 
    in the training set so that the most similar point is found.
     
    :param x_train: series of train vectors
    :param y_train: series of train classes
    :param x_test: series of vectors to be classfied
    :param w: window size
    :return: predictions for test data
    """
    if len(x_train) != len(y_train):
        raise ValueError('x_train and y_train must have same length !')

    predictions = []

    for x in x_test:
        min_dist = np.inf
        closest_class = []
        for i_y, y in enumerate(x_train):
            if dtw_keogh_lower_bound(x, y, 5) < min_dist:
                dist = dtw_window_distance(x, y, w)
                if dist < min_dist:
                    min_dist = dist
                    closest_class = y_train[i_y]

        predictions.append(closest_class)

    if isinstance(x_test, pd.DataFrame):
        predictions = pd.Series(predictions, index=x_test.index)

    return predictions


def spatio_temporal_distance(s1, s2, w):
    f1, f2 = s1[0:2], s2[0:2]
    v1, v2 = s1[2:], s2[2:]

    dis_inv = spatial.distance.euclidean(f1, f2)
    dis_var = dtw_window_distance(v1, v2, w)
    return np.sqrt(dis_inv + dis_var)


def k_means_clustering(data, n_clusters, n_iterations, w=5):
    """
    K-means clustering based on dynamic time warping for time series data
    
    Example:
    -------
    import matplotlib.pylab as plt
    
    train = np.genfromtxt('datasets/train.csv', delimiter='\t')
    test = np.genfromtxt('datasets/test.csv', delimiter='\t')
    data = np.vstack((train[:,:-1],test[:,:-1]))

    centroids,_ = k_means_clust(data,4,10,4)
    for i in centroids:
        plt.plot(i)
    plt.show()
    
    :param data: 
    :param n_clusters: 
    :param n_iterations: 
    :param w: 
    :return: centroids
    """

    if not isinstance(data, list):
        data = list(data)

    centroids = random.sample(data, n_clusters)
    counter = 0
    assignments = {}

    for n in range(n_iterations):
        counter += 1
        assignments = {}

        # assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = np.inf
            closest_clust = None

            for c_ind, j in enumerate(centroids):
                if dtw_keogh_lower_bound(i, j, 5) < min_dist:
                    # cur_dist = dtw_window_distance(i, j, w)
                    cur_dist = spatio_temporal_distance(i, j, w)

                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind

            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0

            if assignments[key]:
                for k in assignments[key]:
                    clust_sum = clust_sum + data[k]
            else:
                clust_sum = [clust_sum]

            centroids[key] = [m / max(len(assignments[key]), 1) for m in clust_sum]

    return centroids, assignments
