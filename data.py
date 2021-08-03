import numpy as np
import jax.numpy as jnp
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

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

def load_mnist(train=True, reshape=True, fashion=False):
    if fashion:
        mnist_dataset = FashionMNIST('/tmp/fashion_mnist/', download=True, train=train)
    else:
        mnist_dataset = MNIST('/tmp/fashion_mnist/', download=True, train=train)
    if reshape:
        images_mnist = jnp.array(mnist_dataset.test_data.numpy().reshape(len(mnist_dataset.test_data), -1), dtype=jnp.float32)
    else:
        images_mnist = jnp.array(np.expand_dims(mnist_dataset.test_data.numpy(), axis=3), dtype=jnp.float32)
    labels_mnist = jnp.array(mnist_dataset.test_labels, dtype=jnp.float32)
    return images_mnist, labels_mnist


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory, 
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))