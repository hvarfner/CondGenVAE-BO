import os
import numpy as np
import jax.numpy as jnp
from torch.utils import data
from torchvision.datasets import MNIST, FashionMNIST


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


IMAGE_SHAPE = 0
TEST_SIZE = 0

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
    IMAGE_SHAPE = 28
    if fashion:
        mnist_dataset = FashionMNIST('/tmp/fashion_mnist/', download=True, train=train)
    else:
        mnist_dataset = MNIST('/tmp/fashion_mnist/', download=True, train=train)
    if reshape:
        images_mnist = jnp.array(mnist_dataset.test_data.numpy().reshape(
            len(mnist_dataset.test_data), -1), dtype=jnp.float32)
    else:
        images_mnist = jnp.array(np.expand_dims(
            mnist_dataset.test_data.numpy(), axis=3), dtype=jnp.float32)
    labels_mnist = jnp.array(mnist_dataset.test_labels, dtype=jnp.float32)
    if not train:
        TEST_SIZE = labels_mnist.shape[0]
    return images_mnist, labels_mnist


def load_dexnet_file(data_f, name, nr=0):
    nr_str = f'{nr:05d}'
    return np.load(data_f + "/" + name + nr_str + '.npz')['arr_0']


def get_num_classes(object_labels):
    return int(object_labels[-1])


def fetch_dexnet_files(data_f, num_classes):
    depth_im_t_f = load_dexnet_file(data_f, 'depth_ims_tf_table_')
    object_labels = load_dexnet_file(data_f, 'object_labels_')
    metric = load_dexnet_file(data_f, 'robust_ferrari_canny_')
    class_nr = 0
    file_nr = 1
    while class_nr <= num_classes:
        depth_im_t_f = np.append(depth_im_t_f, load_dexnet_file(
            data_f, 'depth_ims_tf_table_', file_nr), axis=0)
        object_labels = np.append(object_labels, load_dexnet_file(
            data_f, 'object_labels_', file_nr), axis=0)
        metric = np.append(metric, load_dexnet_file(
            data_f, 'robust_ferrari_canny_', file_nr), axis=0)
        class_nr = get_num_classes(object_labels)
        file_nr = file_nr + 1
    return (depth_im_t_f, object_labels, metric)


def get_class_indices(object_labels, num):
    return np.where(object_labels == num)


def load_dexnet(num_per_class=-1):
    IMAGE_SHAPE = 32
    cwd = os.getcwd()
    dataset_f = os.path.join(cwd, "dataset/")
    data_f = os.path.join(dataset_f, "3dnet_kit_06_13_17")
    if not os.path.exists(data_f):
        print("Dataset does not exist. Downloading it. This will take a while.")
        os.chdir(dataset_f)
        os.system('wget - v - L - O dexnet_2.tar.gz "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=6mnb2bzi5zfa7qpwyn7uq5atb7vbztng&file_id=f_226328650746"')
        os.system("tar - xzf dexnet_2.tar.gz")
        os.system("rm dexnet_2.tar.gz")
        os.chdir(cwd)
    # Load hand-picked classes
    classes_for_learning = np.array([0, 1, 2, 4, 6, 13, 18, 19, 20, 23])
    data = fetch_dexnet_files(data_f, classes_for_learning[-1])
    # Find class with least amount of samples and possibly limit
    min_samples = 10000000
    for i in classes_for_learning:
        if len(get_class_indices(data[1], i)[0]) < min_samples:
            min_samples = len(get_class_indices(data[1], i)[0])
    if num_per_class == -1:
        num_per_class = min_samples
    elif num_per_class > min_samples:
        raise ValueError("Requested more samples per class then the smallest class has samples")
    img_array = np.empty(
        shape=(num_per_class * classes_for_learning.shape[0], data[0][0].shape[0]**2), dtype=object)
    metric_array = np.empty(num_per_class * classes_for_learning.shape[0])
    # Randomly sample from the chosen classes
    rng = np.random.default_rng(12345)
    for i in range(classes_for_learning.shape[0]):
        class_idxs = get_class_indices(data[1], classes_for_learning[i])
        for j in range(num_per_class):
            idx = int(rng.random() * len(class_idxs[0]) + class_idxs[0][0])
            img_array[i * num_per_class + j] = jnp.asarray(data[0][idx].flatten())
            metric_array[i * num_per_class + j] = data[2][idx]
    jax_img_array = jnp.asarray(img_array, dtype=jnp.float32)
    jax_metric_array = jnp.asarray(metric_array, dtype=jnp.float32)
    return jax_img_array, jax_metric_array


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
