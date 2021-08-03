import sys
import os
import json
import pickle
from functools import partial
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import numpy as np
import jax.numpy as jnp
import onnxruntime
import matplotlib.pyplot as plt
from data import load_mnist
from jax.experimental import optimizers
from vae import init_vanilla_vae, mnist_regressor
from vae import quantity_of_interest
from utils import get_best_point
from objective import objective_function, ambigous_objective_function
import matplotlib

matplotlib.use('TkAgg')

if __name__ == '__main__':
    optimize_digit = 3
    number_to_plot = 1
    with open('trained_parameters.pkl', 'rb') as f:
        trained_params = pickle.load(f)
    
    # if wanting to use a fully connected VAE or Convolutional (not yet implemented)
    if len(sys.argv) == 1:
        vae_type = 'vanilla'
    else:
        vae_type = sys.argv[1]
    reshape = vae_type == 'vanilla'
    step_size = 0.001

    best_opt_state = optimizers.pack_optimizer_state(trained_params)
    opt_init, opt_update, get_params = optimizers.momentum(0, mass=0)
    params = get_params(best_opt_state)
    encoder_params, decoder_params, _ = params
    _, encode, _, decode = init_vanilla_vae()
    objective = partial(ambigous_objective_function, 
                        decode=decode,
                        decoder_params=decoder_params,
                        digit=optimize_digit
                )
    
    bounds = [{'name': 'x1', 'type': 'continuous', 'domain': (-3.0, 3.0)},
          {'name': 'x2', 'type': 'continuous', 'domain': (-3.0, 3.0)}]
    bayes_opt = BayesianOptimization(
                                f=objective, 
                                domain=bounds,
                                model_type='GP',
                                acquisition_type ='EI',
                                acquisition_jitter = 0.01,
                                initial_design_numdata = 2,
                                exact_feval=True,
                                verbosity=True)
    bayes_opt.run_optimization(max_iter=50, verbosity=True)
    X, y = bayes_opt.get_evaluations()
    best_points = X[np.argsort(y)[0]]    
    image = decode(decoder_params, X).reshape(28, 28)
    plt.imshow(image)
    bayes_opt.plot_acquisition()
    
    
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