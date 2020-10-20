import numpy as np


class Softmax:
    def __init__(self, input_len, out_nodes):

        self.weights = np.random.randn(input_len, out_nodes)/9
        self.biases = np.zeros(out_nodes)

    def forward(self, input):
        input_len, out_nodes = self.weights.shape

        self.last_input_shape = input.shape

        input = input.flatten()

        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases

        # print('Totals = ', totals)

        self.last_totals = totals

        exponential = np.exp(totals)
        # print("exponentials = ", exponential)
        out = exponential/np.sum(exponential)
        return out

    def backdrop(self, d_L_d_out, learn_rate):

        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            exp_last_totals = np.exp(self.last_totals)
            S = np.sum(exp_last_totals)

            d_out_d_totals = -exp_last_totals*(exp_last_totals[i]/(S**2))
            d_out_d_totals[i] = exp_last_totals[i] * \
                ((S-exp_last_totals[i])/(S**2))

            d_total_d_w = self.last_input
            d_totals_d_b = 1
            d_totals_d_input = self.weights

            num_length = self.last_input.shape[0]
            d_L_d_totals = gradient*d_out_d_totals
            d_L_d_w = np.matmul(d_total_d_w.reshape(
                num_length, 1), d_L_d_totals.reshape(1, 10))
            d_L_d_b = d_L_d_totals*d_totals_d_b
            d_L_d_input = np.matmul(d_totals_d_input, d_L_d_totals)

            self.weights -= learn_rate*d_L_d_w
            self.biases -= learn_rate*d_L_d_b

            return d_L_d_input.reshape(self.last_input_shape)
