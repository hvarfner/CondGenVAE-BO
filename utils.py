import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Constants

TEST_SIZE = 7500
IMAGE_SHAPE = (28, 28)
#IMAGE_SHAPE = (28, 28)
def plot_latent_space(all_samples, values):
    colors = ['b', 'g', 'r', 'c', 'y', 'k', 'm', 'purple', 'orange', 'darkgrey']

    fig = plt.figure(figsize=(16, 9))
    plt.scatter(all_samples[:, 0], all_samples[:, 1], c=values,
                cmap=matplotlib.colors.ListedColormap(colors))

    cb = plt.colorbar()
    cb.set_ticklabels(colors)
    plt.show()
    plt.savefig('latentspace.png')

def get_best_point():
    df = pd.read_csv('mnist_output_samples.csv')
    best_value = np.argmin(df.loc[:, 'Value'])
    best_point = df.iloc[best_value, 0:-2].to_numpy()
    return best_point


def plot_latent_space(model, digit=0):
    pass
