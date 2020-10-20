import numpy as np


class Conv:

    def __init__(self, num_of_filters, size=3):
        self.num_of_filters = num_of_filters
        self.size = size
        self.filters = np.random.randn(num_of_filters, self.size, self.size)/9

    def iterate_regions(self, image):

        height, width = image.shape

        for i in range(height-self.size+1):
            for j in range(width-self.size+1):
                img_region = image[i:i+self.size, j:j+self.size]
                yield i, j, img_region

    def forward(self, input):

        height, width = input.shape
        self.last_input = input
        out = np.zeros((height-self.size+1, width -
                        self.size+1, self.num_of_filters))

        for i, j, img_region in self.iterate_regions(input):
            all_layers = self.filters*img_region
            out[i, j] = np.sum(all_layers, axis=(1, 2))

        return out

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for i, j, img_region in self.iterate_regions(self.last_input):
            for k in range(self.num_of_filters):
                d_L_d_filters[k] += d_L_d_out[i, j, k]*img_region

        self.filters -= learn_rate*d_L_d_filters

        return None
