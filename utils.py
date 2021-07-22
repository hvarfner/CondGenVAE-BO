import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_latent_space(all_samples, values):
    colors = ['b', 'g', 'r', 'c', 'y', 'k', 'm', 'purple', 'orange', 'darkgrey']

    fig = plt.figure(figsize=(16,9))
    plt.scatter(all_samples[:, 0], all_samples[:, 1], c=values, cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    cb.set_ticklabels(colors)
    plt.show()
    plt.savefig('latentspace.png')
