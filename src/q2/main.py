import scipy.io as sio
import numpy as np
from crf import *

DATASET_DIR = "q2dataset.mat"
INSTANCE_DIR = "q2instance.mat"


DATA_SET = sio.loadmat(DATASET_DIR)
def split_data(source):
    train_x = None
    train_y = None
    for i in range(source.shape[1]):
        raw = list(source[0,i])
        x,y = raw[0].reshape(1,3,32), raw[1]
        # print("x shape: ", x.shape)
        if train_x is None:
            train_x = x
            train_y = y
        else:
            train_x = np.vstack((train_x, x))
            train_y = np.vstack((train_y, y))
    print(train_x.shape)
    print(train_x[0,:])
    return train_x, train_y- 1

train_x, train_y = split_data(DATA_SET["trainData"])
test_x, test_y = split_data(DATA_SET["testData"])

INSTANCES = sio.loadmat(INSTANCE_DIR)

SAMPLE_PARAM = INSTANCES["sampleModelParams"]
SAMPLE_THETA = INSTANCES["sampleTheta"]
SAMPLE_X = INSTANCES["sampleX"]
SAMPLE_Y = INSTANCES["sampleY"]

# print("sample_X", SAMPLE_X.shape)
# print("sample y", SAMPLE_Y.shape)
crf = CRFOCR(8,4,0.003)

print(crf.calc_nll(SAMPLE_X, SAMPLE_Y[0,:]))
grad_c, grad_i, grad_p = crf.calc_nll_grad(SAMPLE_X, SAMPLE_Y[0,:])

grad_c = grad_c.reshape(1, grad_c.shape[0])
grad_i = grad_i.reshape(1, 32*26*2)
grad_p = grad_p.reshape(1, 26*26)

grad = np.hstack((grad_c,grad_p, grad_i))
print(np.linalg.norm(grad))

a = crf.SGD(train_x, train_y,test_x[:10,:], test_y)