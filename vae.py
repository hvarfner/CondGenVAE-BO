import os
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, FanOut, Relu, Softplus, Sigmoid
import numpy as np
import tensorflow_probability as tfp
import distrax
from data import load_mnist

LATENT_SIZE = 10
IMAGE_SHAPE = (28, 28)
def gaussian_kl(mu, sigmasq):
    return -0.5 * jnp.sum(1 + jnp.log(sigmasq) - mu**2.0 * sigmasq)


def gaussian_sample(rng, mu, sigmasq):
    return mu + jnp.sqrt(sigmasq) * random.normal(rng, mu.shape)


# TODO change this when doing regression or softmax?
# TODO understand the where fully
def bernoulli_logpdf(logits, x):
    return -jnp.sum(jnp.logaddexp(0., jnp.where(x, -1., 1.) * logits))


def elbo(rng, params, images, beta=1):
    encoder_paramd, decoder_params = params
    mu_z, sigmasq_z = encode(encoder_params, images)
    sample = gaussian_sample(rng, mu_z, sigmasq_z)
    logits = decode(decoder_params, sample)
    return bernoulli_logpdf(logits, images) - beta * gaussian_kl(mu_z, sigma_z)


# TODO create the iwelbo as well
def iwelbo(rng, params, images, n_samples=10):
    pass


def image_sample(rng, params, nrow, ncol):    
    _, decoder_params = params
    code_rng, image_rng = random.split(rng)
    # samples from the standard normal in latent space with shape (nrow * ncol, 10)
    latent_sample = random.normal(code_rng, (nrow * ncol, LATENT_SIZE))
    logits = decode(decoder_params, latent_sample)
    sampled_images = random.bernoulli(img_rng, jnp.logaddexp(0., logits))
    return image_grid(nrow, ncol, sampled_images, IMAGE_SHAPE)


def image_grid(nrow, ncol, image_vectors, image_shape):
  images = iter(image_vectors.reshape((-1,) + image_shape))
  return jnp.vstack([jnp.hstack([next(images).T for _ in range(ncol)][::-1])
                    for _ in range(nrow)]).T


# define the VAE - one of FanOuts is softplus due to non-negative variance
encoder_init, encode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,
    FanOut(2),
    stax.parallel(Dense(LATENT_SIZE), stax.serial(Dense(LATENT_SIZE), Softplus)),
)

decoder_init, decode = stax.serial(
    Dense(512), Relu,
    Dense(512), Relu,
    Dense(np.prod(IMAGE_SHAPE)),
)

if __name__ == '__main__':
    step_size = 0.001
    num_epochs = 100
    batch_size = 32
    nrow, ncol = 10, 10  # sampled image grid size
    test_rng = random.PRNGKey(1)  # fixed prng key for evaluation
    train_images = load_mnist(train=True)
    test_images = load_mnist(train=True)
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)
    print(num_complete_batches, leftover)
    print(num_batches)