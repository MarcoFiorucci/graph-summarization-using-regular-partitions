"""
Coding: UTF-8
Indentation : 4spaces
"""
from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import pandas as pd
import process_datasets as proc_d
import putils as pu
import time
import metrics
import os
from tqdm import tqdm

def create_dir(dir_name):
    """ Utility to create a folder
    :param dir_name: str name of the directory to be created
    :return: None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)




############## Main script code ##############

create_dir("./csv/")

first = True

for n in tqdm([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000], desc="Dimension G"):

        # First graph
        tm = time.time()
        clusters1 = proc_d.random_clusters(n, 8)
        G, GT, labeling = proc_d.custom_cluster_matrix(n, clusters1, 0.3, 1, 0.2, 0)
        tgen = time.time() -tm
        #print(f"TIME: tgen:{tgen:.2f}")

        # Second graph
        clusters2 = proc_d.random_clusters(n, 8)
        G2, GT2, labeling2 = proc_d.custom_cluster_matrix(n, clusters2, 0.6, 1, 0.1, 0)


        # Compression G1
        c = Codec(0.285, 0.285, 1)
        c.verbose = False
        tm = time.time()
        k, epsilon, classes, sze_idx, reg_list, nirr = c.compress(G, 'indeg_guided')
        tcompression1 = time.time() - tm
        #print(f"TIME: tcompression1:{tcompression1:.2f}")

        # Compression G1
        c = Codec(0.285, 0.285, 1)
        c.verbose = False
        tm = time.time()
        k2, epsilon, classes2, sze_idx, reg_list2, nirr2 = c.compress(G2, 'indeg_guided')
        tcompression2 = time.time() - tm
        #print(f"TIME: tcompression2:{tcompression2:.2f}")       

        # Reduced graph
        tm = time.time()
        red = c.reduced_matrix(G, k, epsilon, classes, reg_list)
        red2 = c.reduced_matrix(G2, k2, epsilon, classes2, reg_list2)
        tred = time.time() - tm
        #print(f"TIME: red:{tred:.2f}")

        # Original SD
        tm = time.time()
        sd_o = metrics.spectral_dist(G,G2)
        tsd =  time.time() - tm
        #print(f"TIME: tsd:{tsd:.2f}")

        # Reduced SD
        tm = time.time()
        sd_red = metrics.spectral_dist(red,red2)
        tsd_red = time.time() - tm
        #print(f"TIME: tsd_red:{tsd_red:.2f}")

        # Writes statistics to .csv 
        t_red = tcompression1 + tcompression1 + tsd_red
        t_o = tsd

        if first:
            with open("./csv/time.csv", 'w') as f:
                f.write("n,t_red,t_o,sd_red,sd_o\n")
            first = False

        with open("./csv/time.csv", 'a') as f:
            f.write(f"{n},{t_red:.2f},{t_o:.2f},{sd_red:.4f},{sd_o:.4f}\n")


######  Plot ######
dataset = "./csv/time.csv"

# sd Time plot
df = pd.read_csv(dataset, delimiter=',')

n = df['n']
t_red = df['t_red'].values
t_o = df['t_o'].values

plt.plot(n, t_red, label="1-stage" )
plt.scatter(n, t_red)
plt.plot(n, t_o, label="2-stage" )
plt.scatter(n, t_o)

plt.ylabel("Time seconds")
plt.xlabel("Dimension of graphs")
plt.legend()
plt.grid()
plt.title("SD time performances")

plt.show()

# Noise robustness plot
sd_red = df['sd_red'].values
sd_o = df['sd_o'].values

plt.plot(n, sd_o, label="1-stage" )
plt.scatter(n, sd_o)
plt.plot(n, sd_red, label="2-stage" )
plt.scatter(n, sd_red)

plt.ylabel("Spectral Distance")
plt.xlabel("Dimension of graphs")
plt.legend()
plt.grid()
plt.title("Noise Robustness")

plt.show()

