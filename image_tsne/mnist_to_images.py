from keras.datasets import mnist, fashion_mnist
import scipy.misc
import numpy as np

(digits_im, digits_labels), (_, _) = mnist.load_data()
(fashion_im, fashion_labels), (_, _) = fashion_mnist.load_data()

fused_dataset = np.concatenate([digits_im, fashion_im], axis=0)

for count, im in enumerate(fused_dataset):
    file_name = 'images/{:07d}.png'.format(count+1)
    # print(im.shape)
    # print(file_name)
    scipy.misc.imsave(file_name, im)

