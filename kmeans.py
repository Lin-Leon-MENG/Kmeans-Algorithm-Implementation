import numpy as np
from statistics import mode


def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):
    """
    kmeans clustering implementation with kmeans++ option.
    Returns the centroids and the associated cluster members.
    :param X:
    :param k:
    :param centroids:
    :param tolerance:
    :return:
    """
    # for kmeans++
    if centroids == "kmeans++":
        # initialize the first centroid
        initial_index = np.random.choice(X.shape[0], 1)
        centroids = X[initial_index].reshape(1, -1)  # the first centroid, make it 1 by p
        # pick the rest centroids
        for _ in np.arange(k-1):  # add k-1 more centroids
            # find the next centroid
            min_distances = []  # collect the minimum distances to existing centroids
            for x in X:
                min_d = np.sqrt(np.sum((x - centroids) ** 2, axis=1)).min()  # the minimum distance of point x
                min_distances.append(min_d)
            next_centroid = X[np.argmax(min_distances)].reshape(1, -1)  # the point x with maximum distance
            centroids = np.append(centroids, next_centroid, axis=0)

    # for other case
    else:
        # consider possible duplicate rows
        unique_rows = np.unique(X, axis=0)  # from where choose initial centroids
        # initialize centroids: k by p
        initial_index = np.random.choice(unique_rows.shape[0], k, replace=False)
        centroids = unique_rows[initial_index]  # the initial centroids

    # update centroids until distance reaches tolerance
    norm_distance = 1e999
    while norm_distance > tolerance:
        # initialize clusters: list of k lists
        clusters = [[] for _ in np.arange(k)]
        # assign x to the closest cluster
        for i in np.arange(X.shape[0]):
            cluster_index = np.argmin(np.sqrt(np.sum((X[i] - centroids) ** 2, axis=1)))  # closest cluster
            clusters[cluster_index].append(i)  # put into the cluster accordingly

        # keep trach of the new centroids for each iteration
        new_centroids = np.zeros([k, X.shape[1]])
        # compute the new centroids
        for j in np.arange(k):
            new_centroids[j] = np.mean(X[clusters[j]], axis=0)

        norm_distance = np.sqrt(np.sum((new_centroids - centroids) ** 2))  # update norm distance
        centroids = new_centroids

    return centroids, clusters


def likely_confusion_matrix(y, clusters):
    """
    Compute the confusion matrix given a binary classification results.
    :param y:
    :param clusters:
    :return:
    """
    k = len(clusters)  # number of clusters
    y_clusters = np.array([y[cluster] for cluster in clusters])  # clusters of y
    prediction = [mode(c) for c in y_clusters]  # most common prediction
    matrix = np.zeros([k, k])  # confusion matrix

    # iterate throught each cluster (prediction)
    for j in np.arange(k):
        y_cluster = y_clusters[j]
        cluster_pred = prediction[j]
        # iterate throught each unique y (truth)
        for i in np.unique(y):
            matrix[i][cluster_pred] = np.sum(y_cluster == i)
    matrix = matrix.astype(int)

    # print confusion matrix
    print("       pred F  pred T")
    print("Truth")
    print("F        ", matrix[0][0], "    ", matrix[0][1])
    print("T          ", matrix[1][0], "   ", matrix[1][1])

    # print accuracy
    correct_ct = np.sum(matrix[_][_] for _ in np.arange(k))  # sum diagonal elements
    print("clustering accur", correct_ct / len(y))


def reassign_grey(X, centroids, clusters):
    """
    Reassign the cluster members with the value of its centroid. (For greyscale images)
    :param X:
    :param centroids:
    :param clusters:
    :return:
    """
    X_shape = X.shape  # size of the image
    X_ = X.reshape(-1, 1)  # transform to p = 1 space
    k = len(centroids)  # number of clusters
    # reassign the centroid value to the corresponding cluster
    for c in np.arange(k):
        cluster = clusters[c]
        new_grey = np.array(list(centroids[c]) * len(cluster)).reshape(-1, 1)  # replicate the centroid value
        X_[cluster] = new_grey
    X = X_.reshape(X_shape)  # transform back to original size


def reassign_colors(X, centroids, clusters):
    """
    Reassign the cluster members with the value of its centroid. (For color images)
    :param X:
    :param centroids:
    :param clusters:
    :return:
    """
    X_shape = X.shape
    X_ = X.reshape(-1, 3)  # transform to p = 3 space
    k = len(centroids)
    for c in np.arange(k):
        cluster = clusters[c]
        new_grey = np.array(list(centroids[c]) * len(cluster)).reshape(-1, 3)
        X_[cluster] = new_grey
    X = X_.reshape(X_shape)
