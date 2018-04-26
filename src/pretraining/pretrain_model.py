import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import json
from keras.utils import to_categorical

# GPU compatibility setup
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



# Load ResNet model
feature_extractor = torchvision.models.resnet18(pretrained=False)
# for param in feature_extractor.parameters():
#     param.requires_grad = False

# Remove last layer as it is irrelevant
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-4])


# Now, add multi-objective
class class_color_net(nn.Module):
    def __init__(self):
        super(class_color_net, self).__init__()

        self.in_shape = (3, 28, 28)
        self.upsampler = nn.Upsample(size=(224, 224), mode='bilinear')
        # self.normalize = batch_normalizer
        self.feature_extractor = feature_extractor
        self.extracted_size = self._get_ext_feat_size()
        # Linear output predicts background color so size 3
        self.lin_out_layer = nn.Linear(self.extracted_size, 3)
        # Softmax layer should identify Mnist / Fashion-mnist categories : 20
        self.cat_out_layer = nn.Linear(self.extracted_size, 20)

    def _get_ext_feat_size(self):
        bs = 1
        inpt = Variable(torch.rand(bs, *self.in_shape))
        # inpt = self.normalize(inpt)
        inpt = self.upsampler(inpt)
        output_feat = self.feature_extractor(inpt)
        total_size = output_feat.data.view(bs, -1).size(1)
        return total_size

    def forward(self, x):
        # x = self.normalize(x)
        x = self.upsampler(x)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        rgb_out = self.lin_out_layer(x)
        class_out = self.cat_out_layer(x)
        # class_out = F.softmax(class_out, dim=1)
        return class_out, rgb_out


def sampler(X, Y_1, Y_2, batch_size=128, n_to_sample=16384):
    # Iterator over batches, does not return full dataset
    n_ex = X.shape[0]
    to_sample = np.random.permutation(n_ex)[:n_to_sample]

    tmp = 0

    while tmp < n_to_sample:
        yield (X[to_sample[tmp:tmp+batch_size]], Y_1[to_sample[tmp:tmp+batch_size]],
                    Y_2[to_sample[tmp:tmp+batch_size]])
        tmp += batch_size



def train_model(bi_output_model, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(bi_output_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                bi_output_model.train(True)  # Set model to training mode
            else:
                bi_output_model.train(False)  # Set model to evaluate mode

            running_loss_mse = 0.0
            running_loss_ent = 0.0
            running_corrects = 0

            X = np.load('X_{}.npy'.format(phase), mmap_mode='r')
            Y_class = np.load('Y_{}_class.npy'.format(phase), mmap_mode='r')
            # print(np.bincount(Y_class))
            Y_color = np.load('Y_{}_color.npy'.format(phase), mmap_mode='r')

            n_to_sample = 4096 if phase == 'train' else 1024

            for data in sampler(X, Y_class, Y_color, batch_size=64, n_to_sample=n_to_sample):
                inputs, class_labels_, color_labels_ = data

                color_labels = color_labels_ / 255. - 0.5 # Normalize output for faster convergence
                class_labels = class_labels_


                inputs = FloatTensor(inputs)
                class_labels = LongTensor(class_labels)
                color_labels = FloatTensor(color_labels)

                inputs = Variable(inputs.cuda())
                class_labels = Variable(class_labels.cuda())
                color_labels = Variable(color_labels.cuda())


                pred_scores, pred_colors = bi_output_model(inputs)

                # print(pred_scores.data.cpu().numpy()[0] * (pred_scores.data.cpu().numpy()[0] > 0.01))

                cross_entropy = nn.CrossEntropyLoss()
                mse_loss = nn.MSELoss()

                # CrossEntropyLoss takes integer labels, not one-hot
                cat_loss = cross_entropy(pred_scores, class_labels)
                lin_loss = mse_loss(pred_colors, color_labels)

                loss_seq = [cat_loss, lin_loss]
                grad_seq = [loss_seq[0].data.new(1).fill_(1) for _ in range(len(loss_seq))]

                # backward + optimize only if in training phase
                if phase == 'train':
                    optimizer.zero_grad()
                    torch.autograd.backward(loss_seq, grad_seq)


                    for param in list(filter(lambda p: p.requires_grad, bi_output_model.parameters())):
                        try:
                            param.grad.data.clamp_(-10., 10.)
                        except:
                            pass
                    optimizer.step()

                _, pred_classes = pred_scores.max(1)
                # print(pred_classes.data.cpu().numpy())

                # statistics about color MSE and class accuracy
                running_loss_mse += lin_loss.data[0] * inputs.size(0)
                running_loss_ent += cat_loss.data[0] * inputs.size(0)
                running_corrects += np.sum(pred_classes.data.cpu().numpy() == class_labels.data.cpu().numpy())

            epoch_loss_ent = running_loss_ent / n_to_sample
            epoch_loss_mse = running_loss_mse / n_to_sample
            epoch_acc = running_corrects / n_to_sample

            print('{} Color MSELoss: {:.4f}, XentLoss: {:.4f}, Class acc: {:.4f}'.format(
                        phase, epoch_loss_mse, epoch_loss_ent, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(bi_output_model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    bi_output_model.load_state_dict(best_model_wts)
    torch.save(bi_output_model.state_dict(), './pretrained_model.pth')

    return bi_output_model

full_model = class_color_net().cuda()

optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, full_model.parameters())), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


train_model(full_model, optimizer, exp_lr_scheduler, num_epochs=25)
