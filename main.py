import sys
import os
import json
import pickle
from functools import partial
from hypermapper import optimizer
import numpy as np
import jax.numpy as jnp
import onnxruntime
import matplotlib.pyplot as plt
from data import load_mnist
from jax.experimental import optimizers
from vae import init_vanilla_vae, mnist_regressor
from vae import quantity_of_interest
from objective import objective_function
import matplotlib
matplotlib.use('TkAgg')

if __name__ == '__main__':
    optimize_digit = 1

    with open('trained_parameters.pkl', 'rb') as f:
        trained_params = pickle.load(f)
    
    # if wanting to use a fully connected VAE or Convolutional (not yet implemented)
    if len(sys.argv) == 1:
        vae_type = 'vanilla'
    else:
        vae_type = sys.argv[1]
    reshape = vae_type == 'vanilla'
    step_size = 0.001
    # how many points are sampled before the VAE is retrained
    def placeholder_classifier(point):
        return np.random.uniform(low=0, high=np.sum(point), size=10)
    
    best_opt_state = optimizers.pack_optimizer_state(trained_params)
    opt_init, opt_update, get_params = optimizers.momentum(0, mass=0)
    params = get_params(best_opt_state)
    encoder_params, decoder_params, _ = params
    _, encode, _, decode = init_vanilla_vae()
    objective = partial(objective_function, 
                        decode=decode,
                        decoder_params=decoder_params,
                        digit=optimize_digit
                )
    optimizer.optimize('objective/mnist_scenario.json', objective)
    X = np.array(decode(decoder_params, np.array([-3.998724537364795,-3.8874919146778435]).reshape(-1, 2)).reshape(28, 28)) 
    plt.imshow(X)
    plt.show()
    
    
    
    # TODO
    # TODO
    # Train the VAE & Regressor - can probably do like 10:ish dimensions for MNIST
    # This is the previous source of data - the key is to find new stuff that score high on whatever metric
    # Translation - train on the entire training set of grippable objects
    # Fit the surrogate to all the points and their score on objective function - Probably a BNN model or SMAC due to dimensionality?
    # Optimize and decode new point from latent space - see what the new object looks like (essientially if it's a feasible object)
    # Score the new object on how grippable it is
    # Iterate until we find the most grippable object

    # 