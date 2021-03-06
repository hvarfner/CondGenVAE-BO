import os
import numpy as np
from jax import vmap
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


IMAGE_SHAPE = (28, 28)
TEST_SIZE = 2000

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
    global IMAGE_SHAPE, TEST_SIZE
    IMAGE_SHAPE = (28, 28)
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


def load_dexnet(train=True, num_samples=-1, given_classes=None, num_classes=0):
    global IMAGE_SHAPE, TEST_SIZE
    IMAGE_SHAPE = (28, 28)
    cwd = os.path.dirname(__file__)
    dataset_f = os.path.join(cwd, "dataset/")
    data_f = os.path.join(dataset_f, "3dnet_kit_06_13_17")
    if not os.path.exists(data_f):
        print("Dataset does not exist. Downloading it. This will take a while.")
        os.chdir(dataset_f)
        os.system('wget - v - L - O dexnet_2.tar.gz "https://app.box.com/index.php?rm=box_download_shared_file&shared_name=6mnb2bzi5zfa7qpwyn7uq5atb7vbztng&file_id=f_226328650746"')
        os.system("tar - xzf dexnet_2.tar.gz")
        os.system("rm dexnet_2.tar.gz")
        os.chdir(cwd)
    if given_classes is not None:
        num_classes = given_classes[-1]
        print(f'Retrieving class {num_classes}')
        data = fetch_dexnet_files(data_f, num_classes)
        num_classes = given_classes.shape[0]
        # Find class with least amount of samples and possibly limit
        min_samples = 10000000
        for i in given_classes:
            if len(get_class_indices(data[1], i)[0]) < min_samples:
                min_samples = len(get_class_indices(data[1], i)[0])
        if num_samples == -1:
            num_per_class = min_samples * given_classes.shape[0]
        elif num_samples > min_samples * given_classes.shape[0]:
            raise ValueError(
                "Requested more samples per class then the smallest class has samples:", min_samples)
        else:
            num_per_class = int(num_samples / given_classes.shape[0])
    else:
        given_classes = np.arange(0, num_classes)
        num_per_class = int(num_samples / num_classes)
    metric_array = np.empty(num_per_class * num_classes)
    img_array = np.empty(
        shape=(num_per_class * given_classes.shape[0], data[0][0].shape[0]**2), dtype=object)
    # Randomly sample from the chosen classes
    if train:
        rng = np.random.default_rng(12345)
    else:
        rng = np.random.default_rng(54321)
        TEST_SIZE = num_samples
    for i in range(given_classes.shape[0]):
        class_idxs = get_class_indices(data[1], given_classes[i])
        for j in range(num_per_class):
            # Misses handling of empty classes
            idx = int(rng.random() * len(class_idxs[0]) + class_idxs[0][0])
            img_array[i * num_per_class + j] = jnp.asarray(data[0][idx].flatten())
            metric_array[i * num_per_class + j] = i
    img_array_min, img_array_max = img_array.min(), img_array.max()

    #print(f'Number of nan values before minmax:{np.isnan(img_array).sum()}')
        
    #img_array = (img_array - img_array_min) / (img_array_max - img_array_min)
    #print(f'Number of nan after before minmax:{np.isnan(img_array).sum()}')
    metric_array = metric_array / 9
    jax_img_array = jnp.asarray(img_array, dtype=jnp.float32)
    
    def per_example_minmax(arr):
        arrmin, arrmax = np.min(arr), np.max(arr)

        new_arr = (arr - arrmin) / (arrmax - arrmin)
        new_arr = new_arr.reshape(32, 32)[2:30, 2:30].reshape(-1)
        return 1 - new_arr, arrmax - arrmin

    norm_array, arrdiff = vmap(per_example_minmax, 0)(jax_img_array) 
    #print(f'Number of nan after vectorize:{jnp.isnan(norm_array).sum()}')
    non_flat_images = arrdiff > 1e-2

    jax_metric_array = jnp.asarray(metric_array, dtype=jnp.float32)
    
    return jnp.array(norm_array[non_flat_images], dtype=jnp.float32), jax_metric_array[non_flat_images]


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


def load_dexnet_per_class(classes=[]):
    cwd = os.path.dirname(__file__)
    dataset_f = os.path.join(cwd, "dataset/")
    data_f = os.path.join(dataset_f, "3dnet_kit_06_13_17")
    file_nbr = 0    
    all_files = os.listdir(data_f)
    image_files = [file_ for file_ in all_files if 'depth_ims_tf_table_' in file_]
    label_files = [file_ for file_ in all_files if 'object_labels_' in file_]
    count_per_class = np.zeros(2000)
    iteration = 0
    all_images = np.array([])
    all_labels = np.array([])
    for image_file, label_file in zip(image_files, label_files):
        iteration += 1
        if iteration % int(len(label_files) / 10) == 0:
            print('Gone through {} out of {} files.'.format(iteration, len(label_files)))
        image_data = np.load(os.path.join(data_f, image_file))['arr_0']
        label_data = np.load(os.path.join(data_f, label_file))['arr_0']
        correct_labels = np.isin(label_data, classes)
        try:
            image_data = image_data[correct_labels]
            label_data = label_data[correct_labels]
        except:
            print(image_data.shape, label_data.shape)
            print('NOT THE SAME SIZE ')
            continue

        if len(image_data) > 0:
            img_array_min, img_array_max = image_data.min(), image_data.max()

            jax_img_array = jnp.asarray(image_data, dtype=jnp.float32)

            def per_example_minmax(arr):
                arrmin, arrmax = np.min(arr), np.max(arr)

                new_arr = (arr - arrmin) / (arrmax - arrmin)
                new_arr = new_arr.reshape(32, 32)[2:30, 2:30].reshape(-1)
                return 1 - new_arr, arrmax - arrmin

            norm_array, arrdiff = vmap(per_example_minmax, 0)(jax_img_array) 
            non_flat_images = arrdiff > 1e-2
            image_data = np.array(norm_array[non_flat_images])
            label_data  = label_data[np.array(non_flat_images)]

            if len(all_images) == 0:
                all_images = image_data
                all_labels = label_data
            else:    

                all_images = np.append(all_images, image_data, axis=0)
                all_labels = np.append(all_labels, label_data)
                print(f'Added a file of {len(label_data)} examples.')

            uniques, counts = np.unique(label_data, return_counts=True)
            count_per_class[uniques.astype(int)] += counts
    return count_per_class, all_images, all_labels


def load_saved_dexnet():
    train_images = np.load(os.path.join(os.path.dirname(__file__), 'dataset/train_images.npy'))
    test_images = np.load(os.path.join(os.path.dirname(__file__), 'dataset/test_images.npy'))
    train_labels = np.load(os.path.join(os.path.dirname(__file__), 'dataset/train_labels.npy'))
    test_labels = np.load(os.path.join(os.path.dirname(__file__), 'dataset/test_labels.npy'))
    return train_images, test_images, train_labels, test_labels