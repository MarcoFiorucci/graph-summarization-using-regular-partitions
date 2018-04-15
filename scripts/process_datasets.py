"""
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""
import numpy as np
import pandas as pd


def graph_from_points(x, sigma, to_remove=0):
    """ Generates a graph (weighted graph) from a set of points x (nxd) and a sigma decay
    :param x: a numpy matrix of n times d dimension
    :param sigma: a sigma for the gaussian kernel
    :param to_remove: imbalances the last cluster
    :return: a weighted symmetric graph
    """

    n = x.shape[0]
    n -= to_remove
    w_graph = np.zeros((n,n), dtype='float32')

    for i in range(0,n):
        copy = np.tile(np.array(x[i, :]), (i+1, 1))
        difference = copy - x[0:i+1, :]
        column = np.exp(-sigma*(difference**2).sum(1))

        #w_graph[0:i+1, i] = column
        w_graph[0:i, i] = column[:-1] # set diagonal to 0 the resulting graph is different

    return w_graph + w_graph.T


def get_data(path, sigma):
    """ Given a .csv features:label it returns the dataset modified with a gaussian kernel
    :param name: the path to the .csv
    :param sigma: sigma of the gaussian kernel
    :return: np.array((n,n), dtype=float32) GT int8, labels uint32
    """
    df = pd.read_csv(path, delimiter=',', header=None)
    labels = df.iloc[:,-1].astype('category').cat.codes.values
    features = df.values[:,:-1].astype('float32')

    unq_labels, unq_counts = np.unique(labels, return_counts=True)

    G = graph_from_points(features, sigma)
    aux, GT, aux2 = custom_cluster_matrix(len(labels), unq_counts, 0, 0, 0, 0)

    return G.astype('float32'), GT, labels


def synthetic_regular_partition(k, epsilon):
    """ Generates a synthetic regular partition.
    :param k: the cardinality of the partition
    :param epsilon: the epsilon parameter to calculate the number of irregular pairs
    :return: a weighted symmetric graph
    """

    # Generate a kxk matrix where each element is between (0,1]
    mat = np.tril(1-np.random.random((k, k)), -1)

    x = np.tril_indices_from(mat, -1)[0]
    y = np.tril_indices_from(mat, -1)[1]

    # Generate a random number between 0 and epsilon*k**2 (number of irregular pairs)
    n_irr_pairs = round(np.random.uniform(0, epsilon*(k**2)))

    # Select the indices of the irregular  pairs
    irr_pairs = np.random.choice(len(x), n_irr_pairs)

    mat[(x[irr_pairs],  y[irr_pairs])] = 0

    return mat + mat.T


def synthetic_graph(n, d, datatype):
    """ Generate a graph of n nodes with the given density d
    :param d: float density of the graph
    :return: np.array((n, n), dtype='datatype') graph with density d
    """
    G = np.tril(np.random.random((n, n)) <= d, -1).astype(datatype)
    return G + G.T


def density(G, weighted=True):
    """ Calculates the density of a synthetic graph
    :param G: np.array((n, n)) graph
    :param weighted: bool flag to discriminate weighted case
    :return: float density of the graph
    """
    n = G.shape[0]

    if weighted:
        return G.sum() / (n ** 2)

    e = np.where(G == 1)[0].size / 2
    return e / ((n*(n-1))/2)


def random_clusters(n, num_c):
    """
    Creates a list of num_c int numbers whose sum is n. 
    They represent imbalanced square clusters.
    :param n: int total sum
    :param num_c: int number of clusters
    """

    imb_cluster = np.random.randint(1,num_c+1, size=(num_c,))
    imb_cluster = imb_cluster / imb_cluster.sum()
    imb_cluster = [int(n*dim) for dim in imb_cluster]
    if sum(imb_cluster) != n:
        imb_cluster[-1] = imb_cluster[-1]+(n-sum(imb_cluster))
    return imb_cluster 

def fixed_clusters(n, num_c):
    """
    Creates a list of num_c int numbers whose sum is n. 
    They represent balanced square clusters.
    :param n: int total sum
    :param num_c: int number of clusters
    """

    clusters = [n//num_c] * num_c
    if sum(clusters) != n:
        clusters[-1] = clusters[-1]+(n-sum(clusters))

    assert sum(clusters) == n, "Wrong clusters"
    
    return clusters
    

def custom_cluster_matrix(mat_dim, dims, internoise_lvl, internoise_val, intranoise_lvl, intranoise_value):
    """ Custom noisy matrix
    :param mat_dim : int dimension of the whole graph
    :param dims: list(int) list of cluster dimensions
    :param internoise_lvl : float percentage of noise between clusters
    :param internoise_value : float value of the noise
    :param intranoise_lvl : float percentage of noise within clusters
    :param intranoise_value : float value of the noise

    :returns: np.array((n,n), dtype=float32) G the graph, np.array((n,n), dtype=int8) GT the ground truth, 
              np.array(n, dtype=uint32) labels
    """

    if sum(dims) != mat_dim:
        raise ValueError("The sum of clusters dimensions must be equal to the total number of nodes")

    G = np.tril(np.random.random((mat_dim, mat_dim)).astype('float32') < internoise_lvl, -1).astype('float32')
    G = np.multiply(G, internoise_val)

    GT = np.tril(np.zeros((mat_dim, mat_dim), dtype="int8"), -1)

    x = 0
    for dim in dims:
        cluster = np.tril(np.ones((dim,dim), dtype="float32"), -1)
        mask = np.tril(np.random.random((dim, dim)).astype("float32") < intranoise_lvl, -1)

        if intranoise_value == 0:
            cluster += mask
            indices = (cluster == 2)
            cluster[indices] = 0
        else:
            mask = np.multiply(mask, intranoise_value)
            cluster += mask
            indices = (cluster > 1)
            cluster[indices] = intranoise_value

        G[x:x+dim,x:x+dim]= cluster
        GT[x:x+dim,x:x+dim]= np.tril(np.ones(dim, dtype="int8"), -1)

        x += dim

    return (G + G.T), (GT + GT.T), np.repeat(range(1, len(dims)+1,), dims).astype('uint32')

