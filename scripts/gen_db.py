"""
Coding: UTF-8
Indentation : 4spaces
"""
from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import process_datasets as pd
import putils as pu
import metrics
import os
from tqdm import tqdm
import time
import sys


def create_dir(dir_name):
    """ Utility to create a folder
    :param dir_name: str name of the directory to be created
    :return: None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


############## Main Code ##############

# Database folder creation
create_dir("./db")

# Dimension of the graphs
n = int(sys.argv[1])

# Levels of eta1 and eta2: probability of adding spurious edges 
# and probability of corruption of the structure
internoise_levels = np.arange(0.05, 0.35, 0.05)
intranoise_levels = np.arange(0.05, 0.35, 0.05)
num_cs = [4, 8, 12, 16, 20]


for inter in tqdm(internoise_levels, desc="eta_1"):
    for intra in tqdm(intranoise_levels, desc="eta_2"):
        for num_c in tqdm(num_cs, desc="num_c"):

            #print(f"\n### n:{n} num_c:{num_c} ###")

            #print("Generation of G ...")
            clusters = pd.fixed_clusters(n, num_c)
            G, GT, labels = pd.custom_cluster_matrix(n, clusters, inter, 1, intra, 0)

            # epsilon parameter for the approx Alon et al
            c = Codec(0.285, 0.285, 1)
            c.verbose = False

            #print("Generation of compressed/reduced graph ...")
            tm = time.time()
            k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, "indeg_guided")
            red = c.reduced_matrix(G, k, epsilon, classes, reg_list)
            t_compression = time.time() -tm
            #print(f"s:{t_compression:.2f}")


            # Precomputation of the eigenvalues
            #print("Eigenvalues of G ... ")
            tm = time.time()
            G_eig, aux = metrics.eigs(G)
            t_G_eig = time.time() -tm
            #print(f"s:{t_G_eig:.2f}")

            #print("Eigenvalues of red ... ")
            tm = time.time()
            red_eig, aux = metrics.eigs(red)
            t_red_eig = time.time() -tm
            #print(f"s:{t_red_eig:.2f}")

            # Save in the database the graph, the compressed version, the eigenvalues of the graph
            # and the eigenvalues of the compressed graph
            name = f"{n:05d}_{inter:.2f}_{intra:.2f}_{num_c:02d}"
            np.savez_compressed(f"./db/{name}", G=G, red=red, G_eig=G_eig, red_eig=red_eig, t_compression=t_compression, t_G_eig=t_G_eig, t_red_eig=t_red_eig)

            #print(f"[OK] {name} Saved")

