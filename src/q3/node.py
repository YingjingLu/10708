"""
Reference:
https://github.com/AmazaspShumik/Mixture-Models
@ AmazaspShumik

I borrowed the sfotware artecture of hme model from him
Since I think what his artecture is really scalable and moduarity is good.
I personally did implemented core algorithms.
"""


import numpy as np 
import abc
from scipy.misc import logsumexp 
from model import *
import math
import numpy.linalg as LA


class Node(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, node_pos, x_dim, node_type, num_child, parent = None, bias_term = True,
                                          underflow_tol = 1e-10,
                                          max_iter = 5,
                                          conv_thresh = 1e-6,
                                          stop_learning = 1e-6,
                                          ):
        self.x_dim = x_dim
        self.max_iter = max_iter
        self.conv_thresh = conv_thresh
        self.node_type = node_type
        self.num_children = num_child

        self.log_like_test = 0
        self.learning_thresh = stop_learning
        self.child_node_list = []
        self.parent_node = parent
        self.birth_order = node_pos

    @abc.abstractmethod
    def node_m_step(self):
        pass

    @abc.abstractmethod
    def up_tree_pass(self):
        pass

    @abc.abstractmethod
    def down_tree_pass(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    def has_parent(self):
        return (self.parent_node != None)

    def get_parent_and_birth_order(self, node_list):
        # get the parent of the node and thr number of children to the left

        # return [prent_node, birth_order_num]

        if self.has_parent() == False:
            print("Error, this node does not have parent, Node pos: ", self.birth_order)
            return []

        return [self.parent_node, self.birth_order]

    
        ''' L2 norm of change in parameters of gate model'''
        return self.model.delta_param_norm
        
        

class BaseGaterNode(Node):

    def __init__(self, *args, **kwargs):
        super(BaseGaterNode, self).__init__(*args, **kwargs)
        self.var = 1
        self.v = np.ones((self.num_children, self.x_dim), dtype = np.float64) / (self.num_children* self.x_dim)
        
        self.h = np.ones(self.num_children, dtype = np.float64) / self.num_children
        self.g = np.ones(self.num_children, dtype = np.float64) / self.num_children
        # self.normalizer = np.ones(self.n, dtype = np.float64) / self.n
        self.sigma = 1
        self.node_type = "gate"

    def calc_h(self, x, y):
        return self.calc_p(x,y)

    def self.update_h(self, x, y):
        self.h = self.calc_h(x,y)

    # this is supposed to be calc_h
    def calc_p(self, x, y):
        """
        self.v: (num_child, x_dim)
        x: (num_sample, x_dim)
        """

        # (num_sample, num_child)
        num_sample = x.shape[0]
        h_array = np.zeros((num_sample), dtype = np.shape[0])

        # num_sample, num_child
        g = self.calc_g(x)
        for i in range(self.num_children):
            child = self.child_node_list[i]
            # (num_sample, 1)
            g_children = g[:,i].reshape(1,num_sample)
            # (num_sample,)
            p = child.calc_p(x,y).resahpe(1, num_sample)
            # prod 
            prod = (g_children * p).reshape(num_sample, 1)
            h_matrix [:,i] = prod

        # num_sample. 1
        summation = np.sum(h_matrix, axis = 1)

        # convert to (num_sample,)
        return summation[:,0]

    def calc_var(self, x, y, weights):
        """
         - x: (num_sample, x_dim)
         - y: (num_sample, )
         - weights(self.num_children, )
        """
        if self.parent_node != None:
            assert(weights.shape[0] == self.num_children)
            num_sample = x.shape[0]
            top = 0
            bottom = 0
            for child_index in range(self.num_children):
                weight = weights[child_index]
                child = self.child_node_list[i]
                Ux = child.predict(x)
                diff = y-Ux

                top += weight * diff.dot(diff)*2

                bottom += num_sample * num_sample * weight

            new_var = top/bottom - self.parent_node.var

            return new_var
        else:
            print("top var does not need update")

    def calc_g(self,X):
        """
         - X: (None, 2)  matrix
         - self.v: child, x_dim
         return g: (None, child)
        """

        num_sample = X.shape[0]

        g = X.dot(self.v.transpose())
        g = np.exp(g)
        assert(g.shape == (num_sample,self.num_children), "g in gater node not correct")
        divider = np.sum(g, axis = 1).reshape(g.shape[0], 1)
        g = g / divider

        return g

    def predict(self, X):
        """
         - X: (None, 2)  matrix

         return (None,) array
        """
        assert(self.num_children == len(self.child_node_list), "Gater node child does not match config")

        if self.child_node_list == []:
            print("Error: In Gater Node no children to fetch ")
            return
        """
        for each x calculate g value for each node
        self.v : (child, 2)
        X : (None, 2)

        want g: (None, child)
        """
        num_sample = X.shape[0]

        g = self.calc_g(X)

        # want result: (None,)
        # loop through all the children and use the weight to nultiply the res
        mu = np.zeros((num_sample,1),dtype = np.float64)
        prev_shape = mu.shape
        for i in range(len(self.child_node_list)):
            child = self.child_node_list[i]
            mu_ij = child.predict(X).reshape(num_sample,1)
            g_i = g[:,i].reshape(num_sample,1)

            mu += mu_ij*g_i 
            assert(mu.shape == prev_shape)
        mu_final = mu.reshape(1,num_sample)[0,:]

        return mu_final

    def up_tree_pass(self, x, y):

        # perform up tree pass from all children

        # collect the nodes posteror probability

        # update self's probability


        children_list = self.get_children()

        for index, child_node in enumerate(children_list):
            child_node.up_tree_pass(x, y)

        self.update_h(x,y)

    def down_tree_pass(self, x,y):

        # E step
        h = self.calc_h(x, y)

    def node_m_update(self, H, X):
        self.model.fit(H,X,self.bound_weights)

class ExpertNode(Node):

    def __init__(self, *args, **kwargs):
        super(ExpertNode, self).__init__(*args, **kwargs)
        self.U = np.ones(self.x_dim, dtype = np.float64) / 2
        self.node_type = "expert"

        # (num_sample, )
        self.p = None

    def down_tree_pass(X, Y):
        pass

    def up_tree_pass(self, X, Y):
        pass

    def calc_p(self, x, y):
        """
        return (num_sample, )
        """

        num_sample = x.shape[0]
        num_y = y.shape[0]
        assert(num_sample == num_y, "in calc p and numx, numy differs")
        sigma = self.parent_node.sigma
        # pred (1, None)
        pred = self.U.dot(x)
        # (1, None)
        diff = pred.reshape(1, num_sample) - y.reshape(1, num_y)
        diff = np.square(diff) / 2/ sigma**2
        diff = np.exp(diff) / math.sqrt(2*math.pi)/sigma

        assert(diff.shape == (1,num_sample))

        return diff[0,:]

    def predict(self, X):
        """
         - X: (None, 2)
        return (None,) shape array

        """
        mu_ij = self.U.dot(X.transpose())
        if mu_ij.shape == (X.shape[0],1):
            mu_ij = mu_ij.reshape(1, X.shape[0])
        if mu_ij.shape == (1,X.shape[0]):
            mu_ij = mu_ij[0,:]
        
        assert(mu_ij.shape == (X.shape[0],))
        return mu_ij

    def node_e_step(self, X, Y):
        self.p = self.calc_p(X, Y)

    def node_m_step(self, x, y, weight):
        # perform update on sigma and U through linear regression

        """
         - X: (num_sample, 2)
         - Y: (num_sample)
         - weight: number
        """

        """
        num_sample = X.shape[0]

        x_w = X * (self.p.reshape(num_sample, 1))
        y_w = (Y.reshape(num_sample, 1)) * (self.p.reshape(num_sample, 1))

        theta,r,rank,s = np.linalg.lstsq(x_w,y_w)
        self.U = theta.reshape(1,2)[0,:]
        
        # update parent node shared sigma_i

        # (1, num_sample)
        pred = self.U.dot(x_w.transpose()).reshape(num_sample, 1)

        diff = y-pred

        self.parent_node.var = np.dot(diff, diff) / np.sum(self.p)
        """

        # update U
        # U = sum(hij*xy)/sum(x*x) 
        # sum hij * xy
        top_sum = weight*x.transpose().dot(y).reshape(1,2)
        bottom_sum = weight*np.sum(x*x, axis = 0).reshape(1,2)
        new_U = (top_sum / bottom_sum).reshape(1,2)
        self.U = new_U[0,:]

    def calc_ll(self, x,y):
        return self.calc_p(x,y)









