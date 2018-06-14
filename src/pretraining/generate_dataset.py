from keras.datasets import mnist, fashion_mnist
import numpy as np
import time
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
print('done importing')
try:
    os.makedirs('train')
    for cat in range(20):
        os.makedirs('train/{}'.format(cat))
    os.makedirs('test')
    for cat in range(20):
        os.makedirs('test/{}'.format(cat))
except:
    pass

(X_train_digits, y_train_digits), (X_test_digits, y_test_digits) = mnist.load_data()
(X_train_fashion, y_train_fashion), (X_test_fashion, y_test_fashion) = fashion_mnist.load_data()

X_train = np.concatenate([X_train_digits, X_train_fashion], axis=0)
y_train = np.concatenate([y_train_digits, y_train_fashion + 10], axis=0)

X_test = np.concatenate([X_test_digits, X_test_fashion], axis=0)
y_test = np.concatenate([y_test_digits, y_test_fashion + 10], axis=0)

assert np.min(X_train) == 0
assert np.max(X_train) == 255
assert len(np.unique(y_train)) == 20

num_colors = 3

# Build X_aug to contain 3 copies of each initial image with different colors
Y_train_color = np.random.randint(0, 255, size=(num_colors * X_train.shape[0], 3))
Y_train_class = np.tile(y_train, num_colors)
X_train_aug = np.zeros((num_colors * X_train.shape[0], 3, 28, 28))

for i in tqdm.tqdm(range(num_colors * X_train.shape[0])):
    # First, get the uniform background
    tmp = np.repeat(Y_train_color[i].reshape((3,1)), 28, axis=1)
    tmp = tmp.reshape(tmp.shape + (1,))
    tmp = np.repeat(tmp, 28, axis=2)

    # Then, replace non-dark regions by the images
    img = X_train[i % X_train.shape[0]]
    mask = img>10
    tmp[:, mask] = img[mask]
    tmp = tmp.astype(np.uint8)
    # Dump images, mostly for inspection
    Image.fromarray(tmp.transpose((1,2,0)), 'RGB').save('train/{}/{}.jpg'.format(Y_train_class[i], i))
    X_train_aug[i] = tmp

def batch_normalizer(image_batch):
    obj_mean = [0., 0., 0.]
    obj_std = [1., 1., 1.]

    n_samples = image_batch.shape[0]

    for i in range(n_samples):
        for c in range(3):
            # First, scale to correct variance, then move the mean
            image_batch[i, c] = image_batch[i, c] * (obj_std[c] / image_batch[i, c].std())
            image_batch[i, c] = image_batch[i, c] + obj_mean[c] - image_batch[i, c].mean()
    return image_batch

# X_train_aug = batch_normalizer(X_train_aug)

np.save('X_train.npy', X_train_aug)
np.save('Y_train_class.npy', Y_train_class)
np.save('Y_train_color.npy', Y_train_color)


# Same for test, but only one color for simplicity
Y_test_color = np.random.randint(0, 255, size=(X_test.shape[0], 3))
Y_test_class = np.concatenate((y_test, y_test, y_test))
X_test_aug = np.zeros((X_test.shape[0], 3, 28, 28))

for i in tqdm.tqdm(range(X_test.shape[0])):
    # First, get the uniform background
    tmp = np.repeat(Y_test_color[i].reshape((3,1)), 28, axis=1)
    tmp = tmp.reshape(tmp.shape + (1,))
    tmp = np.repeat(tmp, 28, axis=2)

    # Then, replace non-dark regions by the images
    img = X_test[i]
    mask = img>10
    tmp[:, mask] = img[mask]
    tmp = tmp.astype(np.uint8)
    Image.fromarray(tmp.transpose((1,2,0)), 'RGB').save('test/{}/{}.jpg'.format(Y_test_class[i], i))
    X_test_aug[i] = tmp

# X_test_aug = batch_normalizer(X_test_aug)
np.save('X_test.npy', X_test_aug)
np.save('Y_test_class.npy', Y_test_class)
np.save('Y_test_color.npy', Y_test_color)
