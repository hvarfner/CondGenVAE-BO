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
from data import load_mnist, load_dexnet, load_saved_dexnet
from utils import plot_latent_space
import argparse
IMAGE_SHAPE = (28, 28)
TEST_SIZE = 10000
plot = False
def Reshape(new_shape):
    """Layer construction function for flattening all but the leading dim."""
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0],) + new_shape
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        output_shape = (inputs.shape[0],) + new_shape
        return jnp.reshape(inputs, output_shape)
    return init_fun, apply_fun


def gaussian_kl(mu, sigmasq):
    return -0.5 * jnp.sum(1. + jnp.log(sigmasq) - mu**2. - sigmasq)


def gaussian_sample(rng, mu, sigmasq):
    return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)


# TODO change this when doing regression or softmax?
# TODO understand the where fully
def bernoulli_logpdf(logits, x):
    return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits))


def sample_latent_space(rng, params, images):
    encoder_params, decoder_params, _ = params
    mu_z, sigmasq_z = encode(encoder_params, images)
    sample = gaussian_sample(rng, mu_z, sigmasq_z)
    return sample


def elbo(rng, params, images, beta=1, sep=False):
    encoder_params, decoder_params, _ = params
    mu_z, sigmasq_z = encode(encoder_params, images)
    sample = gaussian_sample(rng, mu_z, sigmasq_z)
    logits = decode(decoder_params, sample)
    if sep:
        return (bernoulli_logpdf(logits, images), -beta * gaussian_kl(mu_z, sigmasq_z), jnp.sum(logits), jnp.sum(images)) 
    return bernoulli_logpdf(logits, images) - beta * gaussian_kl(mu_z, sigmasq_z)


def regression_loss(rng, params, images, labels):
    encoder_params, _, predictor_params = params
    mu_z, sigmasq_z = encode(encoder_params, images)
    samples = gaussian_sample(rng, mu_z, sigmasq_z)
    output = predict(predictor_params, samples, rng=rng, mode='test')
    error = labels - output
    return jnp.mean(jnp.square(error))


def elbo_and_pred_loss(rng, params, images, labels, beta, pred_weight, n_samples=1):
    iwelbo_loss = 0
    for i in range(n_samples):
        sample_rng, predict_rng = random.split(random.fold_in(rng, i))
        encoder_params, decoder_params, predictor_params = params
        mu_z, sigmasq_z = encode(encoder_params, images)
        samples = gaussian_sample(sample_rng, mu_z, sigmasq_z)
        logits = decode(decoder_params, samples)
        iwelbo_loss += bernoulli_logpdf(logits, images) - beta * gaussian_kl(mu_z, sigmasq_z)
    iwelbo_loss /= n_samples
    # MSE loss
    mu_z, sigmasq_z = encode(encoder_params, images)
    samples = gaussian_sample(predict_rng, mu_z, sigmasq_z)
    output = predict(predictor_params, samples, rng=predict_rng)
    error = labels - output

    mse = jnp.mean(jnp.square(error))
    return pred_weight * mse - iwelbo_loss


# TODO create the iwelbo as well
def iwelbo(rng, params, images, n_samples=32):
    iwelbo_loss = 0
    for i in range(n_samples):
        rng = random.PRNGKey(random.fold_in(i))
        encoder_params, decoder_params, _ = params
        mu_z, sigmasq_z = encode(encoder_params, images)
        sample = gaussian_sample(rng, mu_z, sigmasq_z)
        logits = decode(decoder_params, sample)
        iwelbo_loss += bernoulli_logpdf(logits, images) - beta * gaussian_kl(mu_z, sigmasq_z)
    return iwelbo_loss / n_samples


def image_sample(rng, params, nrow, ncol):
    _, decoder_params, _ = params
    code_rng, image_rng = random.split(rng)
    # samples from the standard normal in latent space with shape (nrow * ncol, 10)
    latent_sample = random.normal(code_rng, (nrow * ncol, latent_size))
    logits = decode(decoder_params, latent_sample)
    sampled_images = random.bernoulli(image_rng, jnp.logaddexp(0., logits))
    sampled_images = jnp.logaddexp(0., logits)

    return image_grid(nrow, ncol, sampled_images, IMAGE_SHAPE)


def image_grid(nrow, ncol, image_vectors, image_shape):
    images = iter(image_vectors.reshape((-1,) + image_shape))
    return jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1])
                      for _ in range(nrow)]).T


# define the VAE - one of FanOuts is softplus due to non-negative variance
def init_vanilla_vae(latent_size):
    encoder_init, encode = stax.serial(
        Dense(512), Relu,
        Dense(512), Relu,
        Dense(256), Relu,
        FanOut(2),
        stax.parallel(Dense(latent_size), stax.serial(Dense(latent_size), Softplus)),
    )

    decoder_init, decode = stax.serial(
        Dense(256), Relu,
        Dense(512), Relu,
        Dense(512), Relu,
        Dense(np.prod(IMAGE_SHAPE))
    )
    return encoder_init, encode, decoder_init, decode


# define the VAE - one of FanOuts is softplus due to non-negative variance
def init_vanilla_vae(latent_size):
    encoder_init, encode = stax.serial(
        Dense(64), Relu,
        Dense(64), Relu,
        Dense(32), Relu,
        FanOut(2),
        stax.parallel(Dense(latent_size), stax.serial(Dense(latent_size), Softplus)),
    )

    decoder_init, decode = stax.serial(
        Dense(32), Relu,
        Dense(64), Relu,
        Dense(64), Relu,
        Dense(np.prod(IMAGE_SHAPE))
    )
    return encoder_init, encode, decoder_init, decode

# define the VAE - one of FanOuts is softplus due to non-negative variance
def init_dexnet_vae(latent_size):
    encoder_init, encode = stax.serial(
        Dense(128), Relu,
        Dense(128), Relu,
        Dense(64), Relu,
        FanOut(2),
        stax.parallel(Dense(latent_size), stax.serial(Dense(latent_size), Softplus)),
    )

    decoder_init, decode = stax.serial(
        Dense(64), Relu,
        Dense(128), Relu,
        Dense(128), Relu,
        Dense(np.prod(IMAGE_SHAPE))
    )
    return encoder_init, encode, decoder_init, decode


def mnist_regressor():
    predictor_init, predict = stax.serial(Dense(128), Relu,
                                          Dense(128), Relu,
                                          Dense(1)
                                          )
    return predictor_init, predict


def mnist_classifier():
    predictor_init, predict = stax.serial(Dense(128), Relu,
                                          Dense(128), Relu,
                                          Dense(num_classes),
                                          LogSoftmax
                                          )

    return predictor_init, predict


def predict_image(rng, params, images):
    encoder_params, _, predictor_params = params
    mu_z, sigmasq_z = encode(encoder_params, images)
    samples = gaussian_sample(rng, mu_z, sigmasq_z)
    output = predict(predictor_params, samples, rng=rng, mode='test')

    return output


# TODO - define the quantity_of_interest that we want to optimize
def quantity_of_interest():
    return 0


if __name__ == '__main__':
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

        train_images, test_images, train_labels, test_labels = load_saved_dexnet()
        print(f'Number of training nan values:{jnp.isnan(train_images).sum()}')
        print(f'Number of test nan values:{jnp.isnan(test_images).sum()}')
        print(f'Loaded DexNet with shape of {train_images.shape}.')
        print(f'Loaded DexNet with shape of {test_images.shape}.')
        train_labels = train_labels / 9
        test_labels = test_labels / 9
    else:
        raise ValueError("Unknown dataset.")


    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)

    imfile = os.path.join(os.path.join(os.getcwd(), "tmp/"), dataset + "_vae_{:03d}.png")
    encoder_init_rng, decoder_init_rng, predictor_init_rng = random.split(random.PRNGKey(2), 3)

    if vae_type == 'vanilla':
        define_vae = init_vanilla_vae

        input_shape = (batch_size, np.prod(IMAGE_SHAPE))
    else:
        define_vae = init_conv_vae
        input_shape = (batch_size, ) + IMAGE_SHAPE + (1, )

    encoder_init, encode, decoder_init, decode = define_vae(latent_size)
    regressor = mnist_regressor

    predictor_init, predict = regressor()
    _, encoder_init_params = encoder_init(encoder_init_rng, input_shape)
    _, decoder_init_params = decoder_init(decoder_init_rng, (batch_size, latent_size))
    _, predictor_init_params = predictor_init(predictor_init_rng, (batch_size, latent_size))
    init_params = (encoder_init_params, decoder_init_params, predictor_init_params)
    opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
    train_images = jax.device_put(train_images)
    train_labels = jax.device_put(train_labels.reshape(-1, 1))
    test_images = jax.device_put(test_images[0:TEST_SIZE])
    test_labels = jax.device_put(test_labels[0:TEST_SIZE].reshape(-1, 1))

    print('Train and test are put on device.')

    def split_and_binarize_batch(rng, i, images, labels, binarize):
        i = i % num_batches
        batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
        batch_labels = lax.dynamic_slice_in_dim(labels, i * batch_size, batch_size)

        return batch, batch_labels

    @jit
    def run_epoch(rng, opt_state, images, labels, beta=1, binarize=False):
        def body_fun(i, opt_state):
            elbo_rng, data_rng = random.split(random.fold_in(rng, i))
            batch, batch_labels = split_and_binarize_batch(
                data_rng, i, images, labels, binarize=binarize)

            def loss(params):
                return elbo_and_pred_loss(
                    elbo_rng, params, batch, batch_labels, beta, pred_weight, n_samples=n_samples) / batch_size
            grads = grad(loss)(get_params(opt_state))
            return opt_update(i, grads, opt_state)
        return lax.fori_loop(0, num_batches, body_fun, opt_state)

    @jit
    def evaluate(opt_state, images, labels, binarize=False):
        params = get_params(opt_state)
        elbo_rng, data_rng, image_rng = random.split(test_rng, 3)

        # Relic from when the MNIST images were binarized
        binarized_test = test_images

        test_elbo = elbo(elbo_rng, params, binarized_test, beta=1) / images.shape[0]
        test_mse = regression_loss(elbo_rng, params, binarized_test, labels)
        sampled_images = image_sample(image_rng, params, 6, 1)
        latent_samples = sample_latent_space(elbo_rng, params, images)
        return test_elbo, test_mse, sampled_images, latent_samples

    opt_state = opt_init(init_params)
    beta_schedule = np.linspace(beta_init, beta_final, num_epochs)
    print(train_images.shape)
    _, counts = np.unique(train_labels, return_counts=True)
    _, counts = np.unique(test_labels, return_counts=True)
    
    print(counts)
    if True:
        img_examples = np.arange(0, 70000, 13000)
        fig, axes = plt.subplots(1, (len(img_examples)), figsize=(20, 6))
        for ax, train_ex in zip(axes, img_examples):
            ax.imshow(np.array(train_images[train_ex]).reshape(IMAGE_SHAPE), cmap=plt.cm.gray)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        plt.tight_layout()
        plt.show()

    print('Starting training!')
    all_elbos = np.zeros(num_epochs)
    all_kl = np.zeros(num_epochs)
    all_lh = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        tic = time.time()
        opt_state = run_epoch(random.PRNGKey(epoch), opt_state, train_images,
                              train_labels, beta=beta_schedule[epoch], binarize=binarize)
        test_elbo, test_mse, sampled_images, latent_samples = evaluate(
            opt_state, test_images, test_labels, binarize=binarize)
        all_elbos[epoch] = test_elbo
        params = get_params(opt_state)
        bern, kl, logs, imgs = elbo(random.PRNGKey(1), params, test_images, beta=1, sep=True)
        all_kl[epoch] = kl
        all_lh[epoch] = bern
        
        print("Ep. {: 3d} ---- ELBO: {} ---- MSE: {} ---- Time: {:.3f} sec".format(epoch,
              test_elbo, test_mse, time.time() - tic))
        print("Log Likelihood: {} ---- KL: {}, Logits: {}, Images: {}".format(bern, kl, logs, imgs))
        plt.imsave(imfile.format(epoch), sampled_images, cmap=plt.cm.gray)
    plt.scatter(latent_samples[:, 0], latent_samples[:, 1], c=test_labels,
                cmap='viridis', s=4)
    cb = plt.colorbar()
    cb.set_ticklabels(list(range(10)))
    plt.show()
    # plot images and their prediction
    plt.plot(np.arange(num_epochs), all_elbos)
    #plt.plot(np.arange(num_epochs), all_kl / all_kl.min())
    #plt.plot(np.arange(num_epochs), all_lh / all_lh.min())
    plt.show()
    params = get_params(opt_state)
    elbo_rng, data_rng, image_rng = random.split(test_rng, 3)
    sampled_images = np.random.choice(TEST_SIZE, 20, replace=False)

    trained_params = optimizers.unpack_optimizer_state(opt_state)
    with open(f'models/trained_parameters_{latent_size}_{dataset}.pkl', 'wb') as f:
        print(f'Saving model - trained_parameters_{latent_size}_{dataset}.pkl...')
        pickle.dump(trained_params, f)
    plt.show()

# TODO CHECK IN ENCODE WHAT SHAPE STUFF IS!
