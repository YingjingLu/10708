
import numpy as np 

"""
Theta: length 2366

0: 26: FiC the probability of the character
2626*32: the probability for a character to be true


"""


class CRFOCR(object):
    def __init__(self, img_width, img_height, _lambda ):
        self.img_width = img_width
        self.img_height = img_height
        self._lambda = _lambda

        # f_C the pobability for a character for a hidden state
        self.theta_C = np.zeros(26, dtype = np.float64)
        # f_I the probability for a pixel on a single state
        self.theta_I = np.zeros((2, 26, img_width*img_height), dtype = np.float64)
        # f_P the probability for adjacent pair operation
        self.theta_P = np.zeros((26,26), dtype = np.float64)

        print("init theta I as ", self.theta_I.shape)

    def calc_single_sum_theta_C(self, Y):

        return self.theta_C[Y]

    def calc_sum_theta_C(self, Y_arr):
        total_sum = 0
        for elem in Y_arr:
            total_sum += self.calc_single_sum_theta_C(elem)

        return total_sum

    def calc_single_sum_theta_I(self, x, Y):

        _sum = 0
        sub_x = self.theta_I[0,Y,:]
        # print(sub_x)
        mask = (x == 1)
        # print(mask)
        _sum += np.sum(sub_x[mask])

        # calculate those on 2 channel
        sub_x = self.theta_I[1,Y,:]
        mask = (x == 2)
        _sum += np.sum(sub_x[mask])
        return _sum
        

    def calc_sum_theta_I(self, X, Y_arr):
        total_sum = 0
        for i in range(X.shape[0]):
            elem_X = X[i,:]
            for elem_Y in Y_arr:
                total_sum += self.calc_single_sum_theta_I(elem_X, elem_Y)

        return total_sum 

    def calc_single_sum_theta_P(self, Y_pair):
        y1, y2 = Y_pair

        return self.theta_P[y1,y1]

    def calc_sum_theta_P(self, Y):
        num_y = Y.shape[0]
        if num_y<2:
            return 0

        total_sum = 0
        for i in range(num_y - 1):
            Y_pair = (Y[i], Y[i+ 1])
            total_sum += self.calc_single_sum_theta_P(Y_pair)
        return total_sum

    def calc_z(self, x):

        total_sum = 0
        for y1 in range(26):
            for y2 in range(26):
                for y3 in range(26):
                    y = np.array([y1, y2, y3])
                    total_sum += np.exp(self.calc_var_sum(x, y))
        return total_sum


    def calc_var_sum(self, x, y):
        total_sum = 0
        for i in range(y.shape[0]):
            y1 = y[i]
            x1 = x[i,:]
            total_sum += self.calc_single_sum_theta_I(x1,y1)
            total_sum += self.calc_single_sum_theta_C(y1)
            if i < y.shape[0] -1:
                y2 = y[i+1]
                total_sum += self.calc_single_sum_theta_P((y1,y2))
        return total_sum

    def calc_theta_e(self, x):
        """
         - x: 3 * 32
        """
        z = self.calc_z(x)
        delta_theta_C = np.zeros(26, dtype = np.float64)
        delta_theta_I = np.zeros((2, 26, self.img_height*self.img_width), dtype = np.float64)
        delta_theta_P = np.zeros((26, 26), dtype = np.float64)

        for y1 in range(26):
            for y2 in range(26):
                for y3 in range(26):
                    y = np.array([y1, y2, y3])
                    cond_prob = np.exp(self.calc_var_sum(x, y))/z
                    # print("Cnd Prob: ", cond_prob)
                    for i in range(y.shape[0]):

                        # theta_C
                        delta_theta_C[y[i]] += cond_prob

                        # theta_I layer 1
                        mask = x[i,:]
                        mask = (mask == 1) * cond_prob 
                        delta_theta_I[0,y[i],:] += mask

                        # theta_I layer 2
                        mask = x[i,:]
                        mask = (mask == 2) * cond_prob 
                        delta_theta_I[1,y[i],:] += mask

                        # theta_P
                        if i < (y.shape[0]-1):
                            delta_theta_P[y[i], y[i+1]] += cond_prob


        return delta_theta_C, delta_theta_I, delta_theta_P
    def calc_nll(self, x, y):

        """
        nll = log(Z) - sum( theta * sum(features) ) + lambda*0.5 * theta^2

         - x: 3*32 matrix
         - y: 1*3 array
        """
        nll = 0

        nll += np.log(self.calc_z(x))

        nll -= self.calc_var_sum(x,y)

        sum_weight = 0
        sum_weight += np.sum(np.square(self.theta_P))
        sum_weight += np.sum(np.square(self.theta_I))
        sum_weight += np.sum(np.square(self.theta_C))
        sum_weight *= (self._lambda*0.5)

        nll += sum_weight

        return nll


    def calc_nll_grad(self, x, y):
        # print("theta_C", self.theta_C.shape)
        # print("Theta_I", self.theta_I.shape)
        # print("Theta_P", self.theta_P.shape)
        delta_theta_C = np.zeros(26, dtype = np.float64)
        delta_theta_I = np.zeros((2, 26, self.img_height*self.img_width), dtype = np.float64)
        delta_theta_P = np.zeros((26, 26), dtype = np.float64)
        # E theta:
        d_c, d_i, d_p = self.calc_theta_e(x)
        # E D
        for i in range(y.shape[0]):
            # calc theta_c
            delta_theta_C[y[i]] -= 1
            # calc theta_I layer 1
            mask = x[i,:]
            mask = (mask == 1)
            delta_theta_I[0,y[i],:] -= mask

            # calc theta_I layer 2
            mask = x[i,:]
            mask = (mask == 2)
            delta_theta_I[1,y[i],:] -= mask

            # calc theta_P layer 3
            if i < (y.shape[0]-1):
                delta_theta_P[y[i], y[i+1]] -= 1

        # theta i:
        delta_theta_I += (self.theta_I * self._lambda + d_i)
        delta_theta_P += (self.theta_P * self._lambda + d_p)
        delta_theta_C += (self.theta_C * self._lambda + d_c)

        return delta_theta_C, delta_theta_I, delta_theta_P

    def eval_sum_loss(self, x_test, y_test):
        total_loss = 0
        for i in range(x_test.shape[0]):
            total_loss += self.calc_nll(x_test[i,:], y_test[i,:])

        return total_loss

    def SGD(self, x_batch, y_batch,x_test, y_test, _iter = 1, learning_rate = 0.003):
        steps = x_batch.shape[0]
        loss_list = []
        for i in range(_iter):
            for step in range(steps):
                learning_rate = 1/(1+0.05*step)
                delta_theta_C, delta_theta_I, delta_theta_P = self.calc_nll_grad(x_batch[step,:], y_batch[step,:])
                self.theta_C -= delta_theta_C * learning_rate
                self.theta_I -= delta_theta_I * learning_rate
                self.theta_P -= delta_theta_P * learning_rate
                nll = self.eval_sum_loss(x_test, y_test) / x_test.shape[0]
                print("NLL: ", nll)
                loss_list.append(nll)

        return loss_list




