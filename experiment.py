import numpy as np
import tensorflow as tf

from server import Server
from client import Client


def get_datasets(num_clients, model_type, dataset_type):
    # Load training and eval data
    if model_type == 'perceptron':
        mnist = tf.contrib.learn.datasets.load_dataset("mnist")
        X_train = np.concatenate((mnist.train.images, mnist.validation.images))
        y_train = np.concatenate((
                    np.asarray(mnist.train.labels, dtype=np.int32),
                    np.asarray(mnist.validation.labels, dtype=np.int32),
                    ))
        X_test = mnist.test.images
        y_test = np.asarray(mnist.test.labels, dtype=np.int32)

        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
    else:
        raise ValueError('Model type {0} not supported.'.format(model_type))

    # Partition data
    if dataset_type == 'iid':
        # Shuffle data (train and test)
        indices = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[indices], y_train[indices]
        indices = np.random.permutation(X_test.shape[0])
        X_test, y_test = X_test[indices], y_test[indices]

        # Partition data
        X_train_list = np.split(X_train, num_clients)
        y_train_list = np.split(y_train, num_clients)
    else:
        raise ValueError('Dataset type {0} not supported.'.format(dataset_type))

    return X_train_list, y_train_list, X_test, y_test
