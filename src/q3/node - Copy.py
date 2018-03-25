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


class Node(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, n, node_pos, k, m, node_type, parent = None, bias_term = True,
                                          underflow_tol = 1e-10,
                                          max_iter = 5,
                                          conv_thresh = 1e-6,
                                          stop_learning = 1e-6,
                                          ):
        self.weights = np.ones(n, dtype = np.float64) / n
        self.bound_weights = np.zeros(n, dtype = np.float64)
        self.node_position = node_pos
        self.k = k
        self.underflow_tol = underflow_tol
        self.m = m
        self.bias = bias_term
        self.max_iter = max_iter
        self.conv_thresh = conv_thresh
        self.n = n
        self.node_type = node_type

        self.log_like_test = 0
        self.learning_thresh = 1e-6
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
    def node_prior(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass



    def get_children(self):

        return self.child_node_list

    def has_parent(self):
        return (self.parent_node != None)

    def get_parent_and_birth_order(self, node_list):
        # get the parent of the node and thr number of children to the left

        # return [prent_node, birth_order_num]

        if self.has_parent() == False:
            print("Error, this node does not have parent, Node pos: ", self.node_position)
            return []

        return [self.parent_node, self.birth_order]

    def get_delta_param_norm(self):
        ''' L2 norm of change in parameters of gate model'''
        return self.model.delta_param_norm
        
        
    def get_delta_log_like(self):
        ''' Returns change in likelihood on m-step'''
        return self.model.delta_log_like

class BaseGaterNode(Node):

    def __init__(self, *args, **kwargs):
        super(BaseGaterNode, self).__init__(*args, **kwargs)
        print(self.n, self.k)
        self.v = np.ones((self.n, self.k), dtype = np.float64) / self.n
        self.normalizer = np.ones(self.n, dtype = np.float64) / self.n
        self.node_type = "gate"

    def down_tree_pass(self, X):

        # E step
        if self.has_parent() == True:
            parent, birth_order = self.self.get_parent_and_birth_order()
            self.weights = parent.v[:,birth_order] - parent.normalizer
            self.weights += parent.weights 
        log_H = self.v - np.outer(self.normalizer, np.ones(self.k))
        H = np.exp(log_H)

        self.node_m_step(H, X)

    def predict(self, X):
        self.node_prior(X)

        children = self.get_childrens(nodes)
        n,m = np.shape(X)
        # weighted mean prediction
        mean_prediction = None
        for i,child_node in enumerate(children):
            w  = np.exp(self.v[:,i])
            children_average = child_node.propagate_prediction(X)
            if len(children_average.shape) > 1:
                k = children_average.shape[1]
                w = np.outer(w,np.ones(k))
            if mean_prediction is None:
                mean_prediction = (w * children_average)
            else:
                mean_prediction += (w * children_average)
        return mean_prediction

    def up_tree_pass(self, X):

        self.node_prior(X)
        children_list = self.get_children()

        for index, child_node in enumerate(children_list):
            if child_node.node_type == "expert":
                self.v[:,i] += child_node.weights 
            elif child_node.node_type == "gate":
                self.v[:, i] += logsumexp(child_node.v, axis = 1)

        self.normalizer = logsumexp(self.v, axis = 1)

    def node_m_update(self, H, X):
        self.model.fit(H,X,self.bound_weights)

    def node_prior(self, X):
        probs = self.model.calc_post_log_prob(X)[0]
        self.v = probs


class GaterNodeIRLS(BaseGaterNode):
    '''
    Gate node of Hierarchical Mixture of Experts with softmax transfer function.
    Calculates responsibilities and updates parmameters using weighted softmax regression.
    '''
    
    def __init__(self,*args,**kwargs):
        ''' Initialises gate node '''
        super(GaterNodeIRLS,self).__init__(*args,**kwargs)
        self.model = IRLS(self.learning_thresh, 5)
        self.model.init_params(self.m)
        self.node_type = "gate"

class ExpertNode(Node):

    def __init__(self, *args, **kwargs):
        super(ExpertNode, self).__init__(*args, **kwargs)
        self.model = LinearGaussian(self.learning_thresh, 1)
        self.model.init_params(self.m)
        self.node_type = "expert"

    def down_tree_pass(X, Y):
        parent, birth_order = self.get_parent_and_birth_order(nodes)
        self.weights = parent.responsibilities[:,birth_order] - parent.normaliser
        self.weights += parent.weights


        self.node_m_step(X, Y)

    def up_tree_pass(self, X, Y):

        self.node_prior(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def cal_log_prob(self, X, Y):
        return self.model.calc_post_log_prob(X, Y)[0]

    def node_prior(self, X, Y):
        self.weights = self.model.calc_post_log_prob(X, Y)[0]

    def node_m_step(self, X, Y):
        self.model.fit(Y,X,self.bound_weights)