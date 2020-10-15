# Bacterial Annotation by Learned Representation Of Genes

<!-- badges: start -->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salzberg-lab/Balrog/blob/master/notebooks/Balrog_0.3.2.ipynb)
<!--badges: end -->


## Overview
Balrog is a prokaryotic gene finder based on a Temporal Convolutional Network. We took a data-driven approach to prokaryotic gene finding, relying on the large and diverse collection of already-sequenced genomes. By training a single, universal model of bacterial genes on protein sequences from many different species, we were able to match the sensitivity of current gene finders while reducing the overall number of gene predictions. Balrog does not need to be refit on any new genome.

Preprint available on bioRxiv [here](https://www.biorxiv.org/content/10.1101/2020.09.06.285304v1).

![Balrog](images/balrog.jpg)

## Getting started
Click the "Open in Colab" button above or [click here](https://colab.research.google.com/github/salzberg-lab/Balrog/blob/master/notebooks/Balrog_0.3.2.ipynb) to get started. 

Press the play button on the left side of each cell to run it. Alternatively, hold shift or ctrl and press enter to run cells.
Double click the top of a cell to inspect the code inside and change things. Double click the right side of the cell to hide the code.
Have fun!

Because Balrog uses a complex gene model and performs alignment-based search with mmseqs2, each genome takes ~10-15 minutes to process. Feel free to open a GitHub issue if you run into problems or would like a command line version of Balrog.
