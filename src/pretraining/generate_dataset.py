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

(digits_im, digits_labels), (_, _) = mnist.load_data()
(fashion_im, fashion_labels), (_, _) = fashion_mnist.load_data()
fused_dataset = np.concatenate([digits_im, fashion_im], axis=0)
fused_labels = np.concatenate([digits_labels, fashion_labels + 10], axis=0)

print(fused_dataset.shape, fused_labels.shape)
assert np.min(fused_dataset) == 0
assert np.max(fused_dataset) == 255
assert len(np.unique(fused_labels)) == 20


X_train, X_test, y_train, y_test = train_test_split(fused_dataset, fused_labels,
                                                test_size=0.33, random_state=42)

# Build X_aug to contain 3 copies of each initial image with different colors
Y_train_color = np.random.randint(0, 255, size=(3 * X_train.shape[0], 3))
Y_train_class = np.concatenate((y_train, y_train, y_train))
X_train_aug = np.zeros((3 * X_train.shape[0], 3, 28, 28))

for i in tqdm.tqdm(range(3 * X_train.shape[0])):
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
    obj_mean = [0.485, 0.456, 0.406]
    obj_std = [0.229, 0.224, 0.225]

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
