import sys
import os
import json
import pickle
from functools import partial
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np
from jax.random import PRNGKey
import jax.numpy as jnp
import onnxruntime
import matplotlib.pyplot as plt
from data import load_mnist
from jax.experimental import optimizers
from vae import init_vanilla_vae, mnist_regressor, image_sample
from vae import quantity_of_interest
from utils import get_best_point
from objective import objective_function, brightest_item_objective_function
import matplotlib

matplotlib.use('TkAgg')

if __name__ == '__main__':

    with open('config.json', 'r') as f:
        full_config = json.load(f)
    config = full_config['opt_args']

    latent_size = config['latent_size']
    n_iter = config['num_iterations']
    optimize_digit = 3
    fashion = True
    brightness = 0
    '''
    Label 	Description
    0 	T-shirt/top
    1 	Trouser
    2 	Pullover
    3 	Dress
    4 	Coat
    5 	Sandal
    6 	Shirt
    7 	Sneaker
    8 	Bag
    9 	Ankle boot
    '''
    if fashion:
        with open(f'models/trained_parameters_{latent_size}_fashion.pkl', 'rb') as f:
            trained_params = pickle.load(f)

    else:
        with open(f'models/trained_parameters_{latent_size}.pkl', 'rb') as f:
            trained_params = pickle.load(f)
    

    best_opt_state = optimizers.pack_optimizer_state(trained_params)
    opt_init, opt_update, get_params = optimizers.momentum(0, mass=0)
    params = get_params(best_opt_state)
    encoder_params, decoder_params, _ = params
    _, encode, _, decode = init_vanilla_vae(latent_size)
    objective = partial(brightest_item_objective_function, 
                        decode=decode,
                        decoder_params=decoder_params,
                        digit=optimize_digit, 
                        fashion=fashion,
                        brightness=brightness
                )
    
    bounds = [{'name': f'x{i}', 'type': 'continuous', 'domain': (-3.0, 3.0)}\
         for i in range(latent_size)]
    bayes_opt = BayesianOptimization(
                                f=objective, 
                                domain=bounds,
                                model_type='GP',
                                acquisition_type ='EI',
                                acquisition_jitter = 0.01,
                                initial_design_numdata = 2,
                                exact_feval=True,
                                verbosity=True)
   
    #image_test = decode(decoder_params, np.array([0, 1]).reshape(1, -1)).reshape(28, 28)
    #plt.imshow(image_test)
    #plt.show()
    bayes_opt.run_optimization(max_iter=n_iter, verbosity=True)
    X, y = bayes_opt.get_evaluations()
    best_value = np.argmin(y)
    image = decode(decoder_params, X[best_value]).reshape(28, 28)
    plt.imshow(image, cmap='gray')
    try:
        bayes_opt.plot_acquisition()
    except:
        print('Could not plot due to the number of dimensions.')
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