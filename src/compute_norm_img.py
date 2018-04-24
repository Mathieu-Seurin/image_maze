from keras.datasets import mnist, fashion_mnist
from image_utils import to_rgb_channel_last
import numpy as np
import itertools

import matplotlib.pyplot as plt

def create_big_colored_dataset():

    (digits_im, digits_labels), (_, _) = mnist.load_data()
    (fashion_im, fashion_labels), (_, _) = fashion_mnist.load_data()

    fused_dataset = np.concatenate([digits_im, fashion_im], axis=0)

    black_area = np.where(fused_dataset < 10)

    fused_dataset = to_rgb_channel_last(fused_dataset)/255
    print("mean dataset", fused_dataset.mean(axis=(0,1,2)))

    n_row = 5
    n_col = 4
    num_sample, h, w, channel = fused_dataset.shape

    big_dataset = np.empty((n_row*n_col*num_sample, h, w, channel))
    background = np.ones((3, n_row, n_col))

    background[0, :, :] = np.tile(np.linspace(0, 1, n_col), (n_row, 1))
    background[2, :, :] = np.tile(np.linspace(1, 0, n_row), (n_col, 1)).T

    for count, (line,col) in enumerate(itertools.product(range(n_row), range(n_col))):
        print(count)
        color = background[:, line, col]
        print("color", color)
        current_dataset = np.copy(fused_dataset)
        current_dataset[black_area] = color

        a = current_dataset[10000]
        print("center", a[14,14,:])
        print("corner", a[0,0,:])

        # plt.imshow(a)
        # plt.show()

        big_dataset[num_sample*count:num_sample*(count+1), :, :, :] = current_dataset


    return big_dataset

# plt.imshow(big_dataset[100])
# plt.show()
# plt.imshow(big_dataset[1000])
# plt.show()
# plt.imshow(big_dataset[1000000])
# plt.show()

#mean = np.mean(big_dataset, axis=(0,1,2))
#print(mean)

std = np.std(big_dataset, axis=(0,1,2))
print(std)
