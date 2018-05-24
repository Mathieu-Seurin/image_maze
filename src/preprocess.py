from keras.datasets import mnist, fashion_mnist
import numpy as np
import time
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


# First part is getting the pretrained feature extractor
class CNNExtractor(nn.Module):
    def __init__(self):
        super (CNNExtractor, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.relu1=nn.ELU()
        nn.init.xavier_uniform(self.cnn1.weight)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        self.cnn2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu2=nn.ELU()
        self.dropout=nn.Dropout(0.2)
        nn.init.xavier_uniform(self.cnn2.weight)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        print('Size of the feature map : {}'.format(self.forward(Variable(torch.rand(1,3,28,28))).shape))

    def forward(self,x):
        out=self.cnn1(x)
        out=self.relu1(out)
        out=self.maxpool1(out)
        out=self.dropout(out)
        out=self.cnn2(out)
        out=self.relu2(out)
        out=self.maxpool2(out)
        out=self.dropout(out)
        return out

# Create the extractor, move it to GPU and set to eval mode
feature_extractor = CNNExtractor().cuda()
feature_extractor.train(False)

# Then, load the relevant part of the pretrained model (it contains the last layers)
pretrained_dict = torch.load('pretrained_model.pth')
model_dict = feature_extractor.state_dict()

# Remove part of the key that was used because feature extractor was a submodule
pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items()}
pretrained_dict = {k: v  for k, v in pretrained_dict.items() if k in model_dict}

# Load the new state dict
feature_extractor.load_state_dict(pretrained_dict)


try:
    for dir_ in  ["train", "test"]:
        os.makedirs(dir_)
        for subdir_ in ["maze_images", "obj_images"]:
            os.makedirs(dir_ + '/' + subdir_)
            for bkg in ["with_bkg", "without_bkg"]:
                os.makedirs(dir_ + '/' + subdir_ + '/' + bkg)
                for cat in range(20):
                    os.makedirs(dir_ + '/' + subdir_ + '/' + bkg + '/' + str(cat))
except:
    pass


def normalize_blob(blob):
    # Take a batch of feature maps, and normalize it across each channel

    # First remove the means
    means = blob.mean(dim=3).mean(dim=2).mean(dim=0)
    means = means.unsqueeze(1).repeat(1, 7)
    means = means.unsqueeze(2).repeat(1, 1, 7)
    for i in range(blob.shape[0]):
        blob[i] -= means

    # Sanity check
    assert (blob.mean(dim=3).mean(dim=2).mean(dim=0) < 0.01).all()

    # Then, normalize standard deviations
    stds = blob.permute(1, 0, 2, 3).contiguous().view(32, -1).std(dim=1)
    print('Mean of the channel stds before normalization: {}'.format(stds.mean()))
    stds = stds.unsqueeze(1).repeat(1, 7)
    stds = stds.unsqueeze(2).repeat(1, 1, 7)
    for i in range(blob.shape[0]):
        blob[i] /= stds

    # Sanity check
    assert ((blob.permute(1, 0, 2, 3).contiguous().view(32, -1).std(dim=1) - 1) < 0.01).all()

    return blob


def generate_one_folder(X=None, y=None, dataset_colors=None, folder=None):
    blob = torch.zeros((X.shape[0], 32, 7, 7))
    for i in tqdm.tqdm(range(X.shape[0])):
        # First, get the uniform background
        tmp = np.repeat(dataset_colors[y[i]].reshape((3,1)), 28, axis=1)
        tmp = tmp.reshape(tmp.shape + (1,))
        tmp = np.repeat(tmp, 28, axis=2)

        # Then, replace non-dark regions by the background
        img = X[i]
        mask = img>10
        tmp[:, mask] = img[mask]
        tmp = tmp.astype(np.uint8)
        # Dump image
        Image.fromarray(tmp.transpose((1,2,0)), 'RGB').save(folder + '/{}/{}.jpg'.format(y[i], i))
        # Normalize, extract feature maps, dump them
        tmp = tmp / 255. - 0.5
        fmap = feature_extractor(Variable(FloatTensor(tmp).unsqueeze(0), volatile=True))
        fmap = fmap.data.squeeze(0)
        torch.save(fmap, folder + '/{}/{}.tch'.format(y[i], i))
        blob[i] = fmap

    blob = normalize_blob(blob)
    # for i in tqdm.tqdm(range(blob.shape[0])):
    #     # print(blob[i].shape)
    #     torch.save(blob[i], 'maze_images/{}/{}_normed.tch'.format(Y_train_class[i], i))



# First, get the whole test set (don't use images seen in pretraining)
(_, _), (digits_im, digits_labels) = mnist.load_data()
(_, _), (fashion_im, fashion_labels) = fashion_mnist.load_data()
fused_dataset = np.concatenate([digits_im, fashion_im], axis=0)
fused_labels = np.concatenate([digits_labels, fashion_labels + 10], axis=0)

# Split into images used for train and test
X_train, X_test, y_train, y_test = train_test_split(fused_dataset,
                                fused_labels, test_size=0.33, random_state=42)

# Split again for obj and maze
X_train_maze, X_train_obj, y_train_maze, y_train_obj = train_test_split(X_train,
                                y_train, test_size=0.33, random_state=42)

X_test_maze, X_test_obj, y_test_maze, y_test_obj = train_test_split(X_test,
                                y_test, test_size=0.33, random_state=42)

# Colors as in maze
background = np.ones((3, 5, 4))
background[0, :, :] = np.tile(np.linspace(0, 1, 4), (5, 1))
background[2, :, :] = np.tile(np.linspace(1, 0, 5), (4, 1)).T

maze_colors = [background[:, cat // 4, cat % 4] * 255 for cat in range(20)]
full_dark = [np.array([0,0,0]) for cat in range(20)]

# Train with background
generate_one_folder(X=X_train_maze, y=y_train_maze, dataset_colors=maze_colors,
                    folder='train/maze_images/with_bkg')
generate_one_folder(X=X_train_obj, y=y_train_obj, dataset_colors=maze_colors,
                    folder='train/obj_images/with_bkg')

# Train without background
generate_one_folder(X=X_train_maze, y=y_train_maze, dataset_colors=full_dark,
                    folder='train/maze_images/without_bkg')
generate_one_folder(X=X_train_obj, y=y_train_obj, dataset_colors=full_dark,
                    folder='train/obj_images/without_bkg')

# Test with background
generate_one_folder(X=X_test_maze, y=y_test_maze, dataset_colors=maze_colors,
                    folder='test/maze_images/with_bkg')
generate_one_folder(X=X_test_obj, y=y_test_obj, dataset_colors=maze_colors,
                    folder='test/obj_images/with_bkg')

# Test without background
generate_one_folder(X=X_test_maze, y=y_test_maze, dataset_colors=full_dark,
                    folder='test/maze_images/without_bkg')
generate_one_folder(X=X_test_obj, y=y_test_obj, dataset_colors=full_dark,
                    folder='test/obj_images/without_bkg')
