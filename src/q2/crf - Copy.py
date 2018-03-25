
import numpy as np 

"""
Theta: length 2366

0: 26: FiC the probability of the character
2626*32: the probability for a character to be true


"""


class CRFOCR(object):
    def __init__(self, img_width, img_height, _lambda, ):
        self.img_width = img_width
        self.img_height = img_height
        self._lambda = _lambda

        # f_C the pobability for a character for a hidden state
        self.theta_C = np.zeros(26, dtype = np.float32)
        # f_I the probability for a pixel on a single state
        self.theta_I = np.zeros((img_width*img_height*26*2), dtype = np.float32)
        # f_P the probability for adjacent pair operation
        self.theta_P = np.zeros((26,26), dtype = np.float32)

    def calc_single_sum_theta_C(self, Y):
        """
        mask = np.zeros(26, dtype = np.float32)
        mask[Y] = 1
        _sum = np.sum(mask*self.theta_C)
        """
        return self.theta_C[Y]

    def calc_sum_theta_C(self, Y_arr):
        total_sum = 0
        for elem in Y_arr:
            total_sum += self.calc_single_sum_theta_C(elem)

        return total_sum

    def calc_single_sum_theta_I(self, X, Y):
        """
        mask = np.zeros(self.img_height*self.img_width*26*2, dtype = np.float32)
        factor_index = X*Y
        for index in factor_index:
            mask[index] = 1
        return np.sum(mask*self.theta_I)
        """
        """
         - X: (32,)
         - Y: number
        """

        mask = np.arange(X.shape[0], dtype = np.float32)
        mask = X*mask * Y 
        return np.sum(self.theta_I[mask])

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

    def calc_z(self, x, y):
        # total_sum = 0
        # print("in calc z")
        # print("x shape", x.shape)
        # print("y shape", y)
        # for i in range(y.shape[0]):
        #     part_sum = 0
        #     y1 = y[i]
        #     x1 = x[i,:]
        #     total_sum += self.calc_single_sum_theta_I(x,y1)
        #     total_sum += self.calc_single_sum_theta_C(y1)
        #     if i < y.shape[0] -1:
        #         y2 = y[i+1]
        #         part_sum += self.calc_single_sum_theta_P((y1,y2))

        #     part_sum = np.exp(part_sum)
        #     total_sum += part_sum
        # return total_sum

        """
        given this x with 3*32 
        for all combination 26^3 we calculate the probability for each combination

        """
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
            part_sum = 0
            y1 = y[i]
            x1 = x[i,:]
            total_sum += self.calc_single_sum_theta_I(x,y1)
            total_sum += self.calc_single_sum_theta_C(y1)
            if i < y.shape[0] -1:
                y2 = y[i+1]
                part_sum += self.calc_single_sum_theta_P((y1,y2))

            part_sum = part_sum
            total_sum += part_sum
        return total_sum

    def calc_cond_prob(self, x, y):
        """
         - x: 3 * 32
         - y: 1*3
        """
        feature_weighted_sum = self.calc_var_sum(x, y)
        z = self.calc_z(x, y)
        return np.exp(feature_weighted_sum) / z

    def calc_nll(self, x, y):

        """
        nll = log(Z) - sum( theta * sum(features) ) + lambda*0.5 * theta^2

         - x: 3*32 matrix
         - y: 1*3 array
        """
        nll = 0

        nll += np.log(self.calc_z(x,y))

        nll -= self.calc_var_sum(x,y)

        sum_weight = 0
        sum_weight += np.sum(np.square(self.theta_P))
        sum_weight += np.sum(np.square(self.theta_I))
        sum_weight += np.sum(np.square(self.theta_C))
        sum_weight *= (self._lambda*0.5)

        nll += sum_weight

        return nll

    def calc_cond_prob(self, x, y, y_after):
        z = self.calc_z(x,y)
        _sum = 0
        _sum += self.theta_C[y]
        theta_I_mask = np.zeros(self.img_height*self.img_width*26*2, dtype = np.int)
        for i in range(x.shape[1]):
            theta_I_mask[i*self.img_height*self.img_width*y] = 1

        _sum += np.sum(self.theta_I[theta_I_mask])

        if y_after is not None:
            _sum += self.theta_P[y*y_after]


    def calc_nll_grad(self, x, y):
        delta_theta_C = np.zeros(26, dtype = np.float32)
        delta_theta_I = np.zeros(self.img_height*self.img_width*26*2, dtype = np.float32)
        delta_theta_P = np.zeros(26*26, dtype = np.float32)
        cond_prob = self.calc_cond_prob(x, y)
        for i in range(y.shape[0]):
            x1 = x[i,:].reshape(1, self.img_height*self.img_width)
            y1 = y[i]
            y2 = None
            if i < (y.shape[0]-1):
                y2 = y[i + 1]
            
            if i < (y.shape[0]-1):
                P_mask = np.zeros(26*26, dtype = np.float32)
                P_mask[y1*y2] = self.cond_prob - 1
                delta_theta_P += P_mask
            # E theta
            

            C_mask = np.zeros(26, dtype = np.float32)
            C_mask[y1] = cond_prob - 1
            delta_theta_C = delta_theta_C + C_mask

            I_mask = np.zeros(self.img_height*self.img_width*26*2, dtype = np.float32)
            for i in range(x1.shape[1]):
                index = y1*i*x1[0,i]
                I_mask[index] = cond_prob - 1
            delta_theta_I += I_mask

        # theta i:
        delta_theta_I += self.theta_I * self._lambda
        delta_theta_P += self.theta_P * self._lambda
        delta_theta_C += self.theta_C * self._lambda

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
                self.theta_C += delta_theta_C * learning_rate
                self.theta_I += delta_theta_I * learning_rate
                self.theta_P += delta_theta_P * learning_rate
                loss_list.append(self.eval_sum_loss(x_test, y_test))

        return loss_list




