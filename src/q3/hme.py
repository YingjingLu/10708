"""
I borrowed the sfotware artecture of hme model from:
https://github.com/AmazaspShumik/Mixture-Models/tree/master/Hierarchical%20Mixture%20of%20Experts

Since I think what his artecture is really scalable and moduarity is good.

I personally did implemented core algorithms.
"""

from node import *
import numpy as np 

class HME(object):

    def __init__(self,train_x, train_y, 
                 exp_m, exp_n, verbose = True, max_iter = 100,
                 conv_thresh = 1e-50):

        n = np.shape(train_x)[0]
        train_x = np.concatenate([train_x,np.ones([n,1])], axis = 1)
        self.node_start = None
        self.node_list = []
        self.conv_thresh = conv_thresh
        self.max_iter = max_iter
        self.verbose = verbose

        self.train_x = train_x
        self.train_y = train_y
        self.total_params = 0
        self.delta_param_norm = []
        self.delta_log_like_lb = []
        self.test_log_like = []

        self.n = self.train_x.shape[0]
        self.m = self.train_x.shape[1]

        self.M = exp_m
        self.N = exp_n

        self.construct_artecture()

    def up_tree_pass(self):
        for node in reversed(self.node_list):
            if node.node_type == "expert":
                node.up_tree_pass(self.train_x,self.train_y)
            elif node.node_type == "gate":
                node.up_tree_pass(self.train_x, self.train_y)

    def down_tree_pass(self):

        delta_param_norm = 0
        delta_log_like = 0
        N = len(self.node_list)
        for node in self.nodes:
            if node.node_type == "expert":
                node.down_tree_pass(self.train_x,self.train_y)
            delta_param_norm += node.get_delta_param_norm()
            delta_log_like += node.get_delta_log_like()

        normalised_delta_params = delta_param_norm  / self.total_params
        normalised_delta_like = delta_log_like / self.n*N

        self.delta_param_norm.append(normalised_delta_params)
        self.delta_log_like_lb.append(normalised_delta_like)

    def construct_artecture(self):

        # construct the first layer
        top_node = GaterNodeIRLS(self.n, 0, self.M, self.m, "gate")
        self.node_start = top_node
        self.node_list.append(top_node)

        # in order to ensure the propagation sequence and bring all gater at the front,
        # we cache the experts in sequence at the end of the node list
        expert_node_cache_list = []
        # construct second layer
        for i in range(self.M):
            child_gater = GaterNodeIRLS(self.n, i, self.N, self.m, "gate")
            child_gater.
            self.node_list.append(child_gater)
            top_node.child_node_list.append(child_gater)
            child_gater.parent_node = top_node
            # for the expert one
            for j in range(self.N):
                node = ExpertNode(self.n, j, 1, self.m, "expert")
                child_gater.child_node_list.append(node)
                expert_node_cache_list.append(node)
                node.parent_node = child_gater

        # append all expert node in node list end
        for node in expert_node_cache_list:
            self.node_list.append(node)


    def fit(self):
        for i in range(self.max_iter):
            self.up_tree_pass()
            self.down_tree_pass()
            if self.verbose:
                print("Delta para norm:", self.delta_param_norm[-1])
                print("Delta log like:", self.delta_log_like_lb[-1])
                print("------------------------")
            if self.delta_log_like_lb[-1] <= self.conv_thresh:
                print("Converge")
                return

    def predict(self, X):
        n = np.shape(X)[0]
        X = np.concatenate([X,np.ones([n,1])], axis = 1)
        return self.node_start.predict(X)






        
