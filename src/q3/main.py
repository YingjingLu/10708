import numpy as np 
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import abc
from scipy.stats import norm
import scipy.linalg as LA
from scipy.misc import logsumexp 
from hme import *
from helper import *

data = sio.loadmat("q3_data.mat")
print("dd")

X = data['X']
Y = data['Y']
print("FF")
train_x, train_y, test_x, test_y = train_test_split(X, Y, 0.4)

hme = HME(train_x, train_y, 2, 2, verbose = True,max_iter = 100)
hme.fit()