# graph-summarization-using-regular-partitions

This repository contains a Python 3.6 implementation of a Graph Summarization framework based on Szemerédi's Regulairty Lemma for the task of separating structure from noise in large graphs, as described in:

Marco Fiorucci, Francesco Pelosin and Marcello Pelillo. Separating Structure from Noise in Large Graphs Using the Regularity Lemma. *Pattern Recognition (under review, 2019)*

## Cite

Please cite our paper if you use this code in your own work:
```
@article{Fiorucci2020,
title = "Separating Structure from Noise in Large Graphs Using the Regularity Lemma",
journal = "Pattern Recognition",
volume = "98",
pages = "107070",
year = "2020",
author = "Marco Fiorucci and Francesco Pelosin and Marcello Pelillo"
}
```

## Installation

The packages required are in the file `requirements.txt`

We suggest to create a `virtualenv` and install the packages by just running `pip install -r requirements.txt`.

## Usage

To replicate the experiments in the paper just run `sh experiment.sh` if you are in a Windows system you can just sequentially run the commands specified in the latter file.


