# MyMLPlayground

Welcome to MyMLPlayground! This repository serves as a playground for experimenting with machine learning algorithms, models, datasets, and more.
The main resource for my ML lerning journey has been [Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow (3rd edition)](https://homl.info/er3)

## Overview

In this repository, you'll find:

- **Notebooks**: Jupyter notebooks containing various machine learning experiments and projects.
- **Datasets**: Sample datasets used for training and testing machine learning models.
- **Scripts**: Python scripts for various machine learning tasks, such as data preprocessing, model training, and evaluation.
- **Documentation**: Documentation and tutorials explaining the concepts and techniques used in the machine learning projects.

## Getting Started

### Want to play with these notebooks online without having to install anything?

* <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> (recommended)

⚠ _Colab provides a temporary environment: anything you do will be deleted after a while, so make sure you download any data you care about._

<details>

Other services may work as well, but I have not fully tested them:

* <a href="https://homl.info/kaggle3/"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" /></a>

* <a href="https://mybinder.org/v2/gh/ageron/handson-ml3/HEAD?filepath=%2Findex.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch binder" /></a>

* <a href="https://homl.info/deepnote3/"><img src="https://deepnote.com/buttons/launch-in-deepnote-small.svg" alt="Launch in Deepnote" /></a>

</details>

### Want to run this project using a Docker image?
Read the [Docker instructions](https://github.com/ageron/handson-ml3/tree/main/docker).

### Want to install this project on your own machine?

Start by installing [Anaconda](https://www.anaconda.com/products/distribution) (or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)), [git](https://git-scm.com/downloads), and if you have a TensorFlow-compatible GPU, install the [GPU driver](https://www.nvidia.com/Download/index.aspx), as well as the appropriate version of CUDA and cuDNN (see TensorFlow's documentation for more details).

Next, clone this project by opening a terminal and typing the following commands (do not type the first `$` signs on each line, they just indicate that these are terminal commands):

    $ git clone https://github.com/ageron/handson-ml3.git
    $ cd handson-ml3

Next, run the following commands:

    $ conda env create -f environment.yml
    $ conda activate homl3
    $ python -m ipykernel install --user --name=python3

Finally, start Jupyter:

    $ jupyter notebook

If you need further instructions, read the [detailed installation instructions](INSTALL.md).

## Index

### Notebooks
1. The Machine Learning landscape
2. End-to-end Machine Learning project
3. Classification
4. Training Models
5. Support Vector Machines
6. Decision Trees
7. Ensemble Learning and Random Forests
8. Dimensionality Reduction
9. Unsupervised Learning Techniques
10. Artificial Neural Nets with Keras
11. Training Deep Neural Networks
12. Custom Models and Training with TensorFlow
13. Loading and Preprocessing Data
14. Deep Computer Vision Using Convolutional Neural Networks
15. Processing Sequences Using RNNs and CNNs
16. Natural Language Processing with RNNs and Attention
17. Autoencoders, GANs, and Diffusion Models
18. Reinforcement Learning
19. Training and Deploying TensorFlow Models at Scale

### Scientific Python tutorials
- NumPy
- Matplotlib
- Pandas

### Math Tutorials
- Linear Algebra
- Differential Calculus

### Extra Material
- Auto-differentiation
- s&p 500 market prediction
- simple mnist nn from scratch numpy no tf keras
