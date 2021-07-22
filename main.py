import sys
import os
from vae import *
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, Softplus,\
     Sigmoid, Conv, BatchNorm, Flatten, ConvTranspose, Softmax
from jax.random import multivariate_normal
from data import numpy_collate, NumpyLoader, FlattenAndCast, load_mnist

if __name__ == '__main__':
    vae_type = sys.argv[1]
    reshape = vae_type == 'vanilla'
    beta = 0.10
    pred_weight = 20
    n_samples = 16
    step_size = 0.001
    num_epochs = 50
    batch_size = 256
    nrow, ncol = 10, 10  # sampled image grid size
    test_rng = random.PRNGKey(1)  # fixed prng key for evaluation
    train_images, train_labels = load_mnist(train=True, reshape=reshape)
    test_images, test_labels = load_mnist(train=False)
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)
    
    imfile = os.path.join(os.path.join(os.getcwd(), "tmp/"), "mnist_vae_{:03d}.png")
    encoder_init_rng, decoder_init_rng, predictor_init_rng = random.split(random.PRNGKey(2), 3)
    if vae_type == 'vanilla':
        define_vae = init_vanilla_vae
        input_shape = (batch_size, np.prod(IMAGE_SHAPE))
    else:
        define_vae = init_conv_vae
        input_shape = (batch_size, ) + IMAGE_SHAPE + (1, )

    encoder_init, encode, decoder_init, decode = define_vae()
    predictor_init, predict = mnist_regressor()
    _, encoder_init_params = encoder_init(encoder_init_rng, input_shape)
    _, decoder_init_params = decoder_init(decoder_init_rng, (batch_size, LATENT_SIZE))
    _, predictor_params = predictor_init(predictor_init_rng, (batch_size, LATENT_SIZE))
    init_params = (encoder_init_params, decoder_init_params, predictor_params)

    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    train_images = jax.device_put(train_images)
    train_labels = jax.device_put(train_labels)
    test_images = jax.device_put(test_images[0:5000])
    test_labels = jax.device_put(test_labels[0:5000])


    def binarize_batch(rng, i, images, labels):
        i  = i % num_batches
        batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
        batch_labels = lax.dynamic_slice_in_dim(labels, i * batch_size, batch_size)
        return random.bernoulli(rng, batch).astype(jnp.float32), batch_labels


    @jit
    def run_epoch(rng, opt_state, images, labels):
        def body_fun(i, opt_state):
            elbo_rng, data_rng = random.split(random.fold_in(rng, i))
            batch, batch_labels = binarize_batch(data_rng, i, images, labels)
            loss = lambda params: elbo_and_pred_loss(\
                elbo_rng, params, batch, batch_labels, beta=beta, pred_weight=pred_weight, n_samples=n_samples) / batch_size
            grads = grad(loss)(get_params(opt_state))
            return opt_update(i, grads, opt_state)
        return lax.fori_loop(0, num_batches, body_fun, opt_state)


    @jit
    def evaluate(opt_state, images, labels):
        params = get_params(opt_state)
        elbo_rng, data_rng, image_rng = random.split(test_rng, 3)
        binarized_test = random.bernoulli(data_rng, test_images)
        test_elbo = elbo(elbo_rng, params, binarized_test, beta=1) / images.shape[0]
        test_mse = regression_loss(elbo_rng, params, binarized_test, labels)
        sampled_images = image_sample(image_rng, params, nrow, ncol)
        latent_samples = sample_latent_space(elbo_rng, params, images)
        return test_elbo, test_mse, sampled_images, latent_samples


    opt_state = opt_init(init_params)
    for epoch in range(num_epochs):
        tic = time.time()
        opt_state = run_epoch(random.PRNGKey(epoch), opt_state, train_images, train_labels)
        test_elbo, test_mse, sampled_images, latent_samples = evaluate(opt_state, test_images, test_labels)

        print("Ep. {: 3d} ---- ELBO: {} ---- MSE: {} ---- Time: {:.3f} sec".format(epoch, test_elbo, test_mse, time.time() - tic))
        plt.imsave(imfile.format(epoch), sampled_images, cmap=plt.cm.gray)
    plt.scatter(latent_samples[:, 0], latent_samples[:, 1], c=test_labels,\
        cmap='viridis')
    cb = plt.colorbar()
    cb.set_ticklabels(list(range(10)))
    plt.show()
# TODO CHECK IN ENCODE WHAT SHAPE STUFF IS!