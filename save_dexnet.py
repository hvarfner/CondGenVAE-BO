import os
import numpy as np
from data import load_dexnet_per_class


classes = np.array([1999,1449,1450,1467,420])
train_test_split = 0.7
print('Starting to load images from classes {} ...'.format(', '.join(classes.astype(str))))
counts_per_class, images, labels = load_dexnet_per_class(classes)
print('Done loading all images, retrieved set of size {}'.format(images.shape))
split = np.random.choice(len(images), size=len(images), replace=False)
print('Saving...')
split_idx = int(len(images) * train_test_split)
train_images = images[split][0:split_idx]
test_images = images[split][split_idx:]
train_labels = labels[split][0:split_idx]
test_labels = labels[split][split_idx:]
np.save(os.path.join(os.path.dirname(__file__), 'dataset/train_images.npy'), train_images)
np.save(os.path.join(os.path.dirname(__file__), 'dataset/test_images.npy'), test_images)
np.save(os.path.join(os.path.dirname(__file__), 'dataset/train_labels.npy'), train_labels)
np.save(os.path.join(os.path.dirname(__file__), 'dataset/test_labels.npy'), test_labels)
print('Saved')