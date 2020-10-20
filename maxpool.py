import numpy as np


class Maxpool:
    '''
        Size = 2
    '''

    def __init__(self):
        pass

    def iterate_regions(self, image):
        height, width, _ = image.shape
        new_height = height//2
        new_width = width//2
        for i in range(new_height):
            for j in range(new_width):
                img_region = image[i*2:i*2+2, j*2:j*2+2, :]
                yield i, j, img_region

    def forward(self, input):
        height, width, num_of_filters = input.shape
        self.last_input = input
        new_height = height//2
        new_width = width//2
        out = np.zeros((new_height, new_width, num_of_filters))
        for i, j, img_region in self.iterate_regions(input):
            out[i, j] = np.amax(img_region, axis=(0, 1))

        return out

    def backdrop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for i, j, img_region in self.iterate_regions(self.last_input):
            max_values = np.amax(img_region, axis=(0, 1))
            height, width, num_filters = img_region.shape
            for x in range(height):
                for y in range(width):
                    for z in range(num_filters):
                        if img_region[x, y, z] == max_values[z]:
                            d_L_d_input[i*2+x, j*2+y, z] = d_L_d_out[i, j, z]

        return d_L_d_input
