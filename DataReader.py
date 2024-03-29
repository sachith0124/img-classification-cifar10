import os
import pickle
import numpy as np

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    y_train = []
    for i in range(1, 6):
        with open(f'{data_dir}/data_batch_{i}', 'rb') as train_file:
            dict = pickle.load(train_file, encoding='bytes')
        x_train = dict[b'data'] if i == 1 else np.vstack((x_train, dict[b'data']))
        y_train += dict[b'labels']
    y_train = np.array(y_train)

    cifar_test_dir = f'{data_dir}/test_batch'
    with open(cifar_test_dir, 'rb') as test_file:
        dict = pickle.load(test_file, encoding='bytes')
    x_test = dict[b'data']
    y_test = dict[b'labels']
    y_test = np.array(y_test)
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid
