"""
Coding: UTF-8
Author: lakj
Indentation : 4spaces
"""

from codec import Codec
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import process_datasets as pd
import putils as pu
import metrics
import os
from tqdm import tqdm
import time
import random


def create_dir(dir_name):
    """ Utility to create a folder
    :param dir_name: str name of the directory to be created
    :return: None
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


############## Main Code ##############

create_dir("./csv/")
onestage_csv = "./csv/onestage.csv"
twostage_csv = "./csv/twostage.csv"

db_path = "./db/"

# Takes 5 random graphs from the db based on the groups
# i.e creates Q
db = os.listdir(db_path)
query_set = []
num_cs = [4, 8, 12, 16, 20]
for num_c in num_cs:
    suffix = f"{num_c:02d}.npz"
    query_set.append(random.choice([g for g in db if suffix in g]))


# 1-stage performances
with open(onestage_csv, 'w') as f:
    f.write("q,g,sd,t\n")

# For each q_i in query set Q
for q in tqdm(query_set, desc='1-stage q_i'):
    data = np.load(db_path+q)
    G1 = data['G']
    eig1 = data['G_eig']

    # Compute the query for each graph in db: 
    # time and spectral distance
    tm = time.time()
    for g in tqdm(db, desc='db g'):
        data2 = np.load(db_path+g)
        G2 = data2['G']
        eig2 = data2['G_eig']

        if G1.shape[0] > G2.shape[0]:
            sd = metrics.spectral_dist_offline(G1, G2, eig1, eig2)
        else:
            sd = metrics.spectral_dist_offline(G2, G1, eig2, eig1)

        t_sd = time.time() - tm
        t = data['t_G_eig'] + t_sd

        # Write the results in csv
        with open(onestage_csv, 'a') as f:
            f.write(f"{q},{g},{sd},{t}\n")


# 2-stage
with open(twostage_csv, 'w') as f:
    f.write("q,g,sd,t\n")

# For each q_i in query set Q
for q in tqdm(query_set, desc='2-stage q_i'):
    data = np.load(db_path+q)
    red1 = data['red']
    eig1 = data['red_eig']

    # Compute the query for each graph in db: 
    # time and spectral distance
    tm = time.time()
    for g in tqdm(db, desc='db g'):
        data2 = np.load(db_path+g)
        red2 = data2['red']
        eig2 = data2['red_eig']

        if red1.shape[0] > red2.shape[0]:
            sd = metrics.spectral_dist_offline(red1, red2, eig1, eig2)
        else:
            sd = metrics.spectral_dist_offline(red2, red1, eig2, eig1)

        t_sd = time.time() - tm
        t = data['t_compression'] + data['t_red_eig'] + t_sd

        # Write the results in csv
        with open(twostage_csv, 'a') as f:
            f.write(f"{q},{g},{sd},{t}\n")

