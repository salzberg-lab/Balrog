# Bacterial Annotation by Learned Representation Of Genes

<!-- badges: start -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salzberg-lab/Balrog/blob/master/notebooks/Balrog_0.3.1.ipynb)
<!--badges: end -->


## Overview
Balrog is a prokaryotic gene finder based on a Temporal Convolutional Network. We took a data-driven approach to prokaryotic gene finding, relying on the large and diverse collection of already-sequenced genomes. By training a single, universal model of bacterial genes on protein sequences from many different species, we were able to match the sensitivity of current gene finders while reducing the overall number of gene predictions. Balrog does not need to be refit on any new genome.

Paper link forthcoming...

![Balrog](images/balrog.jpg)

## Getting started
Click the "Open in Colab" button above or [click here](https://colab.research.google.com/github/salzberg-lab/Balrog/blob/master/notebooks/Balrog_0.3.1.ipynb) to get started. Because Balrog uses a complex gene model and performs alignment-based search with mmseqs2, each genome takes ~10-15 minutes to process. Feel free to open a GitHub issue if you run into problems or would like a command line version of Balrog.