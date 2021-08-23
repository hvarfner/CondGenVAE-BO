import sys
import os
import time
import pickle
import json
import matplotlib
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, Softplus,\
    Sigmoid, Conv, BatchNorm, Flatten, ConvTranspose, Softmax, Dropout
from jax.random import multivariate_normal
import numpy as np
import tensorflow_probability as tfp
from data import load_mnist, load_dexnet
from utils import plot_latent_space
import argparse


reshape = True
binarize = False
with open('config.json', 'r') as f:
    full_config = json.load(f)
config = full_config['vae_args']

beta_init = config['beta_init']
beta_final = config['beta_final']
pred_weight = config['pred_weight']
n_samples = config['n_samples']
step_size = config['step_size']
num_epochs = config['num_epochs']
batch_size = config['batch_size']
vae_type = config['vae_type']
latent_size = config['latent_size']
mlp_type = config['mlp_type']
dataset = config['dataset']
dataset_size = config['dataset_size']

nrow, ncol = 10, 10  # sampled image grid size
test_rng = random.PRNGKey(1)  # fixed prng key for evaluation
if dataset == "fashion":
    fashion = True
else:
    fashion = False

if dataset == "fashion" or dataset == "mnist":
    train_images, train_labels = load_mnist(train=True, reshape=reshape, fashion=fashion)
    test_images, test_labels = load_mnist(train=False, fashion=fashion)
    train_images = train_images / 255
    test_images = test_images / 255
    train_labels = train_labels / 9
    test_labels = test_labels / 9
    print(f'Loaded {dataset} with shape of {train_images.shape}.')


elif dataset == "dexnet":
    print('Loading DexNet...')
    #classes = np.array([0, 1, 2, 4, 6, 13, 18, 19, 20, 23])
    classes = np.arange(10)
    train_images, train_labels = load_dexnet(train=True, num_samples=int(dataset_size * 0.8), given_classes=classes)
    print(f'Number of training nan values:{jnp.isnan(train_images).sum()}')
    test_images, test_labels = load_dexnet(
        train=False, num_samples=int(dataset_size * 0.2), given_classes=classes)
    print(f'Number of test nan values:{jnp.isnan(test_images).sum()}')
    print(f'Loaded DexNet with shape of {train_images.shape}.')
else:
    raise ValueError("Unknown dataset.")

