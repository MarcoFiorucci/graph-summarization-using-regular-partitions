import numpy as np
from sklearn import metrics
from scipy import stats
from numpy import linalg as LA
import ipdb

def eigs(m):
    """ Takes a square symmetric matrix and returns the eigenvalues and
    eigenvectors of the normalized laplacian 
    :param m: np.array(float) square symmetric matrix
    :returns: (np.array(), np.array()) tuple with the first component being the
    eigenvalues and the second eigenvectors
    """
    D_half = np.diag(m.sum(1)**(-1/2))
    I = np.identity(m.shape[0])
    norm_L = I - (np.matmul(np.matmul(D_half, m), D_half))
    return LA.eigh(norm_L)

def spectral_dist(m1, m2):
    """ Calculates the spectral distance as specified in 
    Network Summarization with Preserved Spectral Properties Jin
    :param m1: np.array(float) the first matrix bigger than m2
    :param m2: np.array(float) the second matrix smaller than m1
    :returns: float the spectral distance
    """
    n = m1.shape[0]
    k = m2.shape[0]
    big, vec1 = eigs(m1)
    small, vec2 = eigs(m2)

    min_sd = 100
    for l in range(1,k+1):
        if l == k:
            #sd = (1/k)*(small - big[:k]).sum()
            #sd = (1/k)*np.sqrt(((small - big[:k])**2).sum())
            sd = (1/k)*(np.absolute(small - big[:k]).sum())
        else:
            #sd = (1/k)*((small[:l] - big[:l]).sum() + (big[-(k-l):] - small[l:]).sum())
            #sd = (1/k)*np.sqrt((((small[:l] - big[:l])**2).sum() + ((big[-(k-l):] - small[l:])**2).sum()))
            sd = (1/k)*((np.absolute(small[:l] - big[:l])).sum() + (np.absolute(big[-(k-l):] - small[l:])).sum())
            #assert np.all(big[:l] <= small[:l]) and np.all(small[:l] <= big[-(k-l):]), f"Eigenerror \n{big[:l]}\n<=\n{small[:l]}\n<=\n{big[-(k-l):]}"
        #print(sd)    

        #print(f"l:{l} sd:{sd:.4f}")

        if sd < min_sd:
            min_sd = sd
        #print(f"\n\n{[round(x,2) for x in big]}\n{[round(x,2) for x in small]} \n→ {sd}")

    return min_sd

def spectral_dist_offline(m1, m2, big, small):
    n = m1.shape[0]
    k = m2.shape[0]

    min_sd = 100
    for l in range(1,k+1):
        if l == k:
            sd = (1/k)*(np.absolute(small - big[:k]).sum())
        else:
            sd = (1/k)*((np.absolute(small[:l] - big[:l])).sum() + (np.absolute(big[-(k-l):] - small[l:])).sum())

        if sd < min_sd:
            min_sd = sd

    return min_sd



def l2(m1, m2):
    """ Computes the normalized L2 distance between two matrices of the same size
    :param m1: np.array
    :param m2: np.array
    :returns: float, l2 norm
    """
    #return np.sqrt(((m2-m1)**2).sum()) / m1.shape[0]
    return np.linalg.norm(m2-m1)/m2.shape[0]


def l1(m1, m2):
    """ Computes the normalized L1 distance between two matrices of the same size
    :param m1: np.array
    :param m2: np.array
    :returns: float, l2 norm
    """
    #return np.abs(m1 - m2).sum()/m2.shape[0]**2
    return np.linalg.norm(m2-m1, ord=1)/m2.shape[0]


def KL_divergence(d1, d2):
    """ Computes thhe kulback liebeler divergence between two vectors.
    It returns a tuple since the kl divergence is not symmetric
    :param m1: np.array()
    :param m1: np.array()
    :returns: np.array(float64) feature vector of measures
    """
    return stats.entropy(d1, d2), stats.entropy(d2, d1)


def ARI_KVS(m, labeling, ks=[5, 7, 9]):
    """ Implements knn voting system clustering, then compares with the correct labeling
    :param m: np.array((n,n))
    :param labeling: np.array(n) correct clustering
    :returns: adjusted random score
    """

    n = len(labeling)

    max_ars = -10

    for k in ks:
        candidates = np.zeros(n, dtype='uint32')
        i = 0
        for row in m:
            max_k_idxs = row.argsort()[-k:]
            aux = row[max_k_idxs] > 0
            k_indices = max_k_idxs[aux]

            if len(k_indices) == 0:
                k_indices = row.argsort()[-1:]

            #candidate_lbl = np.bincount(labeling[k_indices].astype(int)).argmax()
            candidate_lbl = np.bincount(labeling[k_indices]).argmax()
            candidates[i] = candidate_lbl
            i += 1

        ars = metrics.adjusted_rand_score(labeling, candidates)
        if ars > max_ars:
            max_k = k
            max_ars = ars

    return max_ars


def ARI_DS(m, labeling):
    """ Implements Dominant Set clustering, then compares with the correct labeling
    :param graph: reconstructed graph
    :returns: adjusted random score
    """
    clustering = dominant_sets(m)
    return metrics.adjusted_rand_score(clustering, labeling)


def replicator(A, x, inds, tol, max_iter):
    error = tol + 1.0
    count = 0
    while error > tol and count < max_iter:
        x_old = np.copy(x)
        for i in inds:
            x[i] = x_old[i] * (A[i] @ x_old)
        x /= np.sum(x)
        error = np.linalg.norm(x - x_old)
        count += 1
    return x


def dominant_sets(graph_mat, max_k=4, tol=1e-5, max_iter=1000):
    graph_cardinality = graph_mat.shape[0]
    if max_k == 0:
        max_k = graph_cardinality
    clusters = np.zeros(graph_cardinality)
    already_clustered = np.full(graph_cardinality, False, dtype=np.bool)

    for k in range(max_k):
        if graph_cardinality - already_clustered.sum() <= ceil(0.05 * graph_cardinality):
            break
        # 1000 is added to obtain more similar values when x is normalized
        # x = np.random.random_sample(graph_cardinality) + 1000.0
        x = np.full(graph_cardinality, 1.0)
        x[already_clustered] = 0.0
        x /= x.sum()

        y = replicator(graph_mat, x, np.where(~already_clustered)[0], tol, max_iter)
        cluster = np.where(y >= 1.0 / (graph_cardinality * 1.5))[0]
        already_clustered[cluster] = True
        clusters[cluster] = k
    clusters[~already_clustered] = k
    return clusters


# Test code

def test_sub_sd(big, small):
    min_sd = 100
    k = small.size
    for l in range(1,k+1):
        if l == k:
            #sd = (1/k)*(small - big[:k]).sum()
            sd = (1/k)*np.sqrt(((small - big[:k])**2).sum())
        else:
            #sd = (1/k)*((small[:l] - big[:l]).sum() + (big[-(k-l):] - small[l:]).sum())
            sd = (1/k)*np.sqrt((((small[:l] - big[:l])**2).sum() + ((big[-(k-l):] - small[l:])**2).sum()))

        #print(sd)
        print(f"l:{l} sd:{sd:.4f}")

        if sd < min_sd:
            min_sd = sd


def test_sd():
    G = np.ones((9,9))
    G[0,1] = G[1,0] = G[1,2] = G[2,1] = G[0,2] = G[2,0] =  0
    G[3,4] = G[4,3] = G[4,5] = G[5,4] = G[3,5] = G[5,3] =  0
    G[6,7] = G[7,6] = G[7,8] = G[8,7] = G[6,8] = G[8,6] =  0
    np.fill_diagonal(G, 0)

    #import networkx as nx
    #import matplotlib.pyplot as plt
    #nx.draw_circular(nx.from_numpy_array(G))
    #plt.show()

    print(G)
    red = np.array([[0,9,9],[9,0,9],[9,9,0]])
    print(red)
    sd = spectral_dist(G, red)
    print(f"{sd:.4f}")


#test_sd()
#ipdb.set_trace()
#test_sub_sd(np.array([0,0,1,1,1,1,1,1,2]), np.array([0,0,2]))
#test_sub_sd(np.array([0,1,1,1,1,1,1,1.5,1.5]), np.array([0,1.5,1.5]))
#ttest_sub_sd(np.array([0,0,0,1,1,1,1,1,1]), np.array([0,0,0]))
#test_sub_sd(np.array([0,1,1,1,1,1,1,1]),np.array([0,0,0,1]))


