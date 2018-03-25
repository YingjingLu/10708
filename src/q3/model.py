import numpy as np 
from scipy.stats import norm
import scipy.linalg as LA

def log_softmax(theta, X):
    n,m= np.shape(X)
    m,k = np.shape(Theta)
    X_Theta = np.dot(X,Theta)
    norm = logsumexp(X_Theta, axis = 1)
    log_softmax   = (X_Theta.T - norm).T
    return log_softmax


class LinearGaussian(object):

    def __init__(self, threshold, max_iter):
        self.threshold = threshold
        self.theta = None
        self.var = 0
        self.delta_param_norm = 0
        self.deta_log_like = 0
        self.iter = max_iter

    def init_params(self, x_dim):
        self.theta = np.random.normal(0,1,x_dim)
        self.var = 1

    def fit(self, X, Y, weights = None):

        n,m = X.shape

        if weights is None:
            w = np.ones(n)
            weights = w
        else:
            w = np.sqrt(weights)

        X_w = np.transpose(np.transpose(X) * w)
        Y_w = Y*w

        if self.theta is None:
            self.init_params(m)
        theta_tmp = self.theta
        var_tmp = self.var 
        log_likelihood_tmp = self.calc_log_likelihood(X, Y, weights)

        self.theta, _, _, _ = LA.lstsq(X, Y)

        pred_diff = (Y_w - np.dot(X_w,self.theta))
        pred_diff_variance = np.dot(pred_diff,pred_diff)/np.sum(weights)

        # update variance
        self.var = pred_diff_variance
        # recalculate parameters
        log_like_after = self.calc_log_likelihood(X,Y,weights)
        delta_log_like = ( log_like_after - log_like_before)/n

        # undo the update variance if underflow
        if delta_log_like < self.stop_learning:
            self.theta = theta_recovery
            self.var   = var_recovery
            delta_log_like = 0

        # otherwise save change in parameters and likelihood
        theta_delta = self.theta - theta_tmp
        self.delta_param_norm = np.sum(np.dot(theta_delta.T,theta_delta))
        self.delta_log_like = delta_log_like

    def predict(self, X):
        return np.dot(X, self.theta)

    def calc_post_log_prob(self, x, y):
        if self.theta is None:
            self.init_params(x.shape[1])
        # return normal pdf, log pdf
        # assume Gaussian
        u = y - np.dot(x,self.theta)
        log_norm = -1* np.log(np.sqrt(2*np.pi*self.var))
        log_main = -u*u/(2*self.var)
        log_pdf = log_norm + log_main
        prob = np.exp(log_pdf)
        return (log_pdf,prob)

    def calc_log_likelihood(self, x, y, weights = None):
        if weights is None:
            weights = np.ones(x,shape[0])
        log_pdf, prob = self.calc_post_log_prob(x, y)
        loglikelihood = np.sum(weights * log_pdf)

        return log_likelihood


class IRLS(object):
    def __init__(self, threshold, N, max_iter = 5):
        self.threshold = threshold
        self.v = None
        self.N = N
        self.delta_param_norm = 0
        self.deta_log_like = 0
        self.iter = max_iter

    def init_params(self, x_dim):
        # initialize v as a x_
        self.v = np.random.normal(x_dim, self.N)

    def fit(self, X, Y, min_pack = 1e-10):

        x, p = X.shape
        delta = np.array(np.repeat(min_pack, n)).reshape(1, n)
        w =np.ones(n)
        W = np.diag(w)
        self.v = self.update_v(W, X, Y)
        for _ in range(self.iter):
            v_prev = self.v
            raw_w = abs(y-X.dot(self.v)).T 
            w = float(1)/np.maximum(delta, raw_w)
            W = np.diag(w[0])
            self.v = self.update_v(W, X, Y)
            tol = np.sum(abs(v_prev - self.v))
            if tol < self.threshold:
                return self.v 
        return self.v 

    def update_v(self, W, X, Y):
        return np.dot(np.linalg.inv(X.T.dot(W).dot(X)), (X.T.dot(W).dot(Y)))


