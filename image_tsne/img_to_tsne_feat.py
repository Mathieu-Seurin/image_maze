import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.datasets import mnist, fashion_mnist
from keras.models import Model
from sklearn.decomposition import PCA
from scipy.spatial import distance
from tqdm import tqdm

from image_utils import to_rgb_channel_first, to_rgb_channel_last
from scipy.misc import imresize
from sklearn.manifold import TSNE
from PIL import Image

(digits_im, digits_labels), (_, _) = mnist.load_data()
(fashion_im, fashion_labels), (_, _) = fashion_mnist.load_data()

fused_dataset = np.concatenate([digits_im, fashion_im], axis=0)
# To test
#fused_dataset = fused_dataset[:100]


if not os.path.isfile("mnist_vgg_features.npy"):

    model = keras.applications.VGG16(weights='imagenet', include_top=True)
    feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)

    img_size = (224, 224)
    batch_size = 64

    n_img = fused_dataset.shape[0]

    features = np.empty((n_img, 4096))
    batch = np.empty((batch_size, img_size[0], img_size[1], 3))

    for num_img in tqdm(range(n_img)):

        img = fused_dataset[num_img]
        img_number_in_batch = num_img % batch_size

        img = imresize(img,size=img_size)
        img = to_rgb_channel_last(img)
        # x = np.expand_dims(x, axis=0)

        batch[img_number_in_batch] = img

        if img_number_in_batch == batch_size-1:

            batch = preprocess_input(batch)
            features[num_img-batch_size+1:num_img+1, :] = feat_extractor.predict(batch)
            batch = np.empty((batch_size, img_size[0], img_size[1], 3))

        elif num_img == n_img-1 :
            print("last batch")
            batch = batch[:img_number_in_batch+1]
            batch = preprocess_input(batch)
            features[num_img - img_number_in_batch : n_img, :] = feat_extractor.predict(batch)


    np.save("mnist_vgg_features.npy", features)
    print("computing features ok !")
else:
    print("mnist_vgg_features.npy, loading it.")
    features = np.load("mnist_vgg_features.npy")


if not os.path.isfile("pca_mnist_vgg_features.npy"):
    pca = PCA(n_components=300)
    pca.fit(features)
    pca_features = pca.transform(features)
    np.save("pca_mnist_vgg_features.npy", pca_features)
else:
    print("pca_mnist_vgg_features.npy, loading it.")
    pca_features = np.load("pca_mnist_vgg_features.npy")


tsne = TSNE(n_components=2, learning_rate=150, perplexity=20, angle=0.2, verbose=2).fit_transform(pca_features)
np.save("tsne_features.npy", tsne)

tx, ty = tsne[:,0], tsne[:,1]
tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))

width = 4000
height = 3000
max_dim = 100


np.random.shuffle(fused_dataset)

full_image = Image.new('RGBA', (width, height))
for img, x, y in tqdm(zip(fused_dataset[1000], tx, ty)):
    #tile = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
    tile = Image.fromarray(np.uint8(img))
    tile = tile.resize(size=(224,224))

    rs = max(1, tile.width/max_dim, tile.height/max_dim)
    tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
    full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))

plt.figure(figsize = (16,12))
plt.imshow(full_image)
plt.savefig("tsne_final.png")