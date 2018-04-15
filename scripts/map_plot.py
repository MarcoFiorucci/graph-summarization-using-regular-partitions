import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipdb
import scipy.stats as st
import sys
import os


def avg_time(dataset):
    """ Calculates the mean of the time column of a .csv
    :param dataset: string the path to .csv file
    :return: float the mean
    """
    df = pd.read_csv(dataset, delimiter=',')

    return df['t'].values.mean()


def pak(k, bv):
    """ Returns precision @ k of a binary vector bv
    :param k: int index > 0
    :param bv: list int binary list representing the ranking
    :returns: float a number between 0 and 1 representing the precision @ k
    """
    assert k > 0, "pak requires k>0"
    return sum(bv[:k])/k

def avg_precision(bv):
    """ Returns the average precision for a bit value encoding a ranking
    :param bv: list int binary list representing the ranking
    :returns: float a number between 0 and 1 representing the avg precision
    """
    if sum(bv) == 0:
        return 0
    else: 
        return sum([pak(k+1, bv) for k,b in enumerate(bv) if b == 1])/sum(bv)


def MAP(k, dataset):
    """ Returns the MAP fot top k graphs
    :param k: int number of top-k graph to take in consideration
    :param dataset: string path of the .csv file to parse
    :returns: float a number between 0 and 1 representing the MAP of the ranking
    """
    
    values = []

    df = pd.read_csv(dataset, delimiter=',')

    db_set = df['g'].unique()
    query_set = df['q'].unique()

    # For each query graph
    for q in query_set:
        suffix = q[-6:]
        group = [g for g in db_set if suffix in g]

        df_q = df.loc[df['q'] == q].sort_values('sd')
        k_ranking = df_q['g'][:k]

        # For each top k graph
        bit_vec = []
        for g in k_ranking.values:
            if g in group:
                bit_vec.append(1)
            else:
                bit_vec.append(0)

        values.append(avg_precision(bit_vec))
        #print(f"{q} {bit_vec} {values[-1]}")
        #print(f"{k_ranking}")

    return np.array(values).mean()


##### Main script code #####


n = sys.argv[1]

dataset = f"./csv/onestage.csv"
dataset2 = f"./csv/twostage.csv"

t_o = avg_time(dataset)
t_t = avg_time(dataset2)

ks = range(1,10)
map_onestage = []
map_twostage = []
for k in ks:
    map_onestage.append(MAP(k, dataset))
    map_twostage.append(MAP(k, dataset2))

plt.plot(ks, map_onestage, label=f"1-Stage sec:{t_o:.2f}") 
plt.scatter(ks, map_onestage) 
plt.plot(ks, map_twostage, label=f"2-Stage sec:{t_t:.2f}") 
plt.scatter(ks, map_twostage) 

plt.ylabel("MAP")
plt.xlabel("top-k")
plt.xticks(ks)
plt.legend()
plt.grid()
plt.title(f"top-k MAP n:{n}")


plt.show()

