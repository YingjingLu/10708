import numpy as np 
import random

def train_test_split(X, Y, rate):
    num_sample = X.shape[0]

    test_index_set = set(random.sample(range(num_sample), int(num_sample*rate)))
    total_index_set = set(range(num_sample))

    train_index_set = total_index_set - test_index_set
    train_index = list(train_index_set)
    test_index = list(test_index_set)

    train_x = X[train_index,:]
    train_y = Y[train_index, :]

    test_x = X[test_index,:]
    test_y = Y[test_index,:]

    return train_x, train_y, test_x, test_y



