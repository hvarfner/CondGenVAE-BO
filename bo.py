import sys
import os
import json
import pickle
import hypermapper
from data import load_mnist
from jax.experimental import optimizers
from vae import init_vanilla_vae, mnist_regressor
from vae import quantity_of_interest



if __name__ == '__main__':

    with open('objective/mnist_scenario.json', 'r') as f:
        parameters_file = json.load(f)

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
    retrain_every = 0
    best_opt_state = optimizers.pack_optimizer_state(trained_params)
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    params = get_params(best_opt_state)
    _, encode, _, decode = init_vanilla_vae()
    _, predict = mnist_regressor()
    
    train_images, train_labels = load_mnist(train=True, reshape=reshape)
    print(train_images[0])
    train_images = train_images /255
    train_labels = train_labels / 9
    
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