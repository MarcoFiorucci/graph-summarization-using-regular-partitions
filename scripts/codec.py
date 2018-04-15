"""
File: sensitivity_analysis.py
Description: This class performs a sensitivity analysis of the Szemeredi algorithm
Coding: UTF-8
Author: lakj
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as spst
from sklearn import metrics
import ipdb 
import sys
sys.path.insert(1, '../graph_reducer/')
import szemeredi_lemma_builder as slb
import refinement_step as rs
from scipy import ndimage
from math import ceil


class Codec:

    def __init__(self, e1, e2, samples):

        # Find bounds parameters
        self.min_k = 4
        self.fast_search  = True

        # SZE algorithm parameters
        self.kind = "alon"
        self.is_weighted = True
        self.random_initialization = True

        self.drop_edges_between_irregular_pairs = True

        # SZE running parameters
        self.iteration_by_iteration = False
        self.sze_verbose = False
        self.compression = 0.05

        # Reconstruction parameters
        self.indensity_preservation = False

        # Global print
        self.verbose = True

        self.epsilons = np.linspace(e1,e2,samples)


    def run_alg(self, G, epsilon, refinement):
        """ Creates and run the szemeredi algorithm with a particular dataset
        if the partition found is regular, its cardinality, and how the nodes are partitioned
        :param epsilon: float, the epsilon parameter of the algorithm
        :returns: (bool, int, np.array)
        """
        self.srla = slb.generate_szemeredi_reg_lemma_implementation(self.kind, G, epsilon,
                                                                    self.is_weighted, self.random_initialization,
                                                                    refinement, self.drop_edges_between_irregular_pairs)
        return self.srla.run(iteration_by_iteration=self.iteration_by_iteration, verbose=self.sze_verbose, compression_rate=self.compression)


    def compress(self, G, refinement):
        """ Compression phase
        :param G: np.array((n, n)) matrix
        :param refinement: string refinement to be used
        :return: k, epsilon, classes, sze_idx, regularity list, number of irregular pairs
        """
        if self.verbose:
            print(f"[CoDec] Compression...")
        partitions = {}
        for epsilon in self.epsilons:
            regular, k, classes, sze_idx, reg_list , nirr= self.run_alg(G, epsilon, refinement)

            if self.verbose:
                print(f"    {epsilon:.6f} {k} {regular} {sze_idx:.4f}")

            if regular:
                if k in partitions:

                    # We take the partition with lower epsilon if two of the same size are found
                    if epsilon < partitions[k][0]: 
                        partitions[k] = (epsilon, classes, sze_idx, reg_list, nirr)
                else:
                    partitions[k] = (epsilon, classes, sze_idx, reg_list, nirr)

            elif self.fast_search and not regular and k>self.min_k:
                break

        if partitions == {}:
            print(f"[CoDec] NO regular partitions found.")
        else:

            max_idx = -1
            max_k = -1

            for k in partitions.keys():
                #if partitions[k][2] > max_idx:
                if k > max_k:
                    max_k = k
                    max_idx = partitions[k][2]

            k = max_k
            epsilon = partitions[k][0]
            classes = partitions[k][1]
            sze_idx = partitions[k][2]
            reg_list = partitions[k][3]
            nirr = partitions[k][4]
            if self.verbose:
                print(f"[CoDec] Best partition - k:{k} epsilon:{epsilon:.4f} sze_idx:{sze_idx:.4f} irr_pairs:{nirr}")

        return k, epsilon, classes, sze_idx, reg_list, nirr


    def decompress(self, G, thresh, classes, k, regularity_list):
        """ Reconstruct the original matrix from a reduced one.
        :param thres: the edge threshold if the density between two pairs is over it we put an edge
        :param classes: the reduced graph expressed as an array
        :return: a numpy matrix of the size of 

        -----
        [TODO BUG] wrong implementation on weighted case indensity preservation

        """
        print("[CoDec] Decompression --> SZE")
        n = G.shape[0]
        reconstructed_mat = np.zeros((n, n), dtype='float32')
        for r in range(2, k + 1):
            r_nodes = np.where(classes == r)[0]

            for s in range(1, r) if not self.drop_edges_between_irregular_pairs else regularity_list[r - 2]:
                s += 1
                s_nodes = np.where(classes == s)[0]
                bip_sim_mat = G[np.ix_(r_nodes, s_nodes)]
                n = bip_sim_mat.shape[0]
                bip_density = bip_sim_mat.sum() / (n ** 2.0)

                # Put edges if above threshold
                if bip_density > thresh:
                    if self.is_weighted:
                        #if bip density > somethin?
                        reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = bip_density
                        #m = np.tril(np.random.random((r_nodes.size, r_nodes.size)) <= bip_density, -1).astype('float32')
                        #m[m>0] = bip_density
                        #reconstructed_mat[np.ix_(r_nodes, s_nodes)] =  m
                        #reconstructed_mat[np.ix_(s_nodes, r_nodes)] = m.T
                    else:
                        reconstructed_mat[np.ix_(r_nodes, s_nodes)] = reconstructed_mat[np.ix_(s_nodes, r_nodes)] = 1

        # Implements indensity information preservation
        if self.indensity_preservation:
            for c in range(1, k+1):
                indices_c = np.where(classes == c)[0]
                n = len(indices_c)
                max_edges = (n*(n-1))/2
                n_edges = np.tril(G[np.ix_(indices_c, indices_c)], -1).sum()
                indensity = n_edges / max_edges
                if np.random.uniform(0,1,1) <= indensity:
                    if self.is_weighted:
                        # [TODO BUG] wrong implementation
                        reconstructed_mat[np.ix_(indices_c, indices_c)] = indensity
                    else:
                        erg = np.tril(np.random.random((n, n)) <= indensity, -1).astype('float32')
                        erg += erg.T
                        reconstructed_mat[np.ix_(indices_c, indices_c)] = erg

        np.fill_diagonal(reconstructed_mat, 0.0)
        return reconstructed_mat


    def post_decompression(self, sze, ksize):
        """ Post decompression filtering phase.
        :param sze: np.array((n, n)) decompressed matrix
        :param ksize: int size of kernel
        :return: np.array((n, n)) filtered decompressed matrix
        """
        print("[CoDec] Post-Decompression Filtering SZE --> FSZE")
        fsze = ndimage.median_filter(sze,ksize)
        fsze = np.tril(fsze, -1)
        return fsze + fsze.T 


    def reduced_matrix(self, G, k, epsilon, classes, regularity_list):
        """
        Generates the similarity matrix of the current classes
        :return sim_mat: the reduced similarity matrix
        """
        reduced_sim_mat = np.zeros((k, k), dtype='float32')

        for r in range(2, k + 1):

            r_indices = np.where(classes == r)[0]

            for s in (range(1, r) if not self.drop_edges_between_irregular_pairs else regularity_list[r - 2]):

                s_indices = np.where(classes == s)[0]

                bip_adj_mat = G[np.ix_(s_indices, r_indices)]
                classes_n = bip_adj_mat.shape[0]
                bip_density = bip_adj_mat.sum() / (classes_n ** 2.0)
                reduced_sim_mat[r - 1, s - 1] = reduced_sim_mat[s - 1, r - 1] = bip_density

        return reduced_sim_mat



