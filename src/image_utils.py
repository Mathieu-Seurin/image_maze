import numpy as np
from skvideo.io import FFmpegWriter as VideoWriter
import matplotlib.pyplot as plt
import os

def to_rgb_channel_first(im):

    if im.ndim == 3: # Multiple images
        b, w, h = im.shape
        rgb_im = np.empty((b, 3, w, h), dtype=np.float32)
        rgb_im[:, 0, :, :] = im
        rgb_im[:, 1, :, :] = rgb_im[:, 2, :, :] = rgb_im[:, 0, :, :]

    elif im.ndim == 2:
        w, h = im.shape
        rgb_im = np.empty((3, w, h), dtype=np.float32)
        rgb_im[0, :, :] = im
        rgb_im[1, :, :] = rgb_im[2, :, :] = rgb_im[0, :, :]

    else:
        assert False, "Can only convert images or batches of images\nInput dimension can be (height, width) or (batch, height, width)"

    return rgb_im

def to_rgb_channel_last(im):

    if im.ndim == 3: # Multiple images in batches
        b, w, h = im.shape
        rgb_im = np.empty((b, w, h, 3), dtype=np.float32)
        rgb_im[:, :, :, 0] = im
        rgb_im[:, :, :, 1] = rgb_im[:, :, :, 2] = rgb_im[:, :, :, 0]

    elif im.ndim == 2:
        w, h = im.shape
        rgb_im = np.empty((w, h, 3), dtype=np.float32)
        rgb_im[:, :, 0] = im
        rgb_im[:, :, 1] = rgb_im[:, :, 2] = rgb_im[:, :, 0]
    else:
        assert False, "Can only convert images or batches of images\nInput dimension can be (height, width) or (batch, height, width)"

    return rgb_im

def channel_first_to_channel_last(im):
    assert im.shape[0] == 3, "Image needs to be in format (3,H,W)"
    # swap from c, h, w => h,w,c
    im = im.swapaxes(0, 2)
    im = im.swapaxes(0, 1)
    return im

def channel_last_to_channel_first(im):
    assert im.shape[2] == 3, "Image needs to be in format (H,W,3)"
    # swap from h,w,c => c,h,w
    im = im.swapaxes(0, 2)
    im = im.swapaxes(1, 2)
    return im

def plot_single_image(im):

    if im.shape[0] == 3 :#channel first
        im = channel_first_to_channel_last(im)

    plt.figure()
    plt.imshow(im)
    plt.show()


def make_video(replay, filename):
    n_frames = len(replay)
    n_channels, n_w, n_h = replay[0].shape
    if not os.path.isdir(filename):
        os.makedirs(filename)
    # writer = VideoWriter(filename + '.mp4')
    for i in range(n_frames):
        for _ in range(5):
            # Frame repeat for easier watching
            plt.imshow(replay[i].transpose(1, 2, 0))
            plt.savefig('{}/{}.png'.format(filename, i))
            plt.close()

def make_eval_plot(filename_in, filename_out):
    plop = np.loadtxt(filename_in)
    plop = plop.reshape((-1, 2))
    plt.figure()
    plt.plot(plop[:, 0], plop[:, 1])
    plt.savefig(filename_out)
    plt.close()


def batch_normalizer(image_batch):
    # Takes a batch (n_samples, 3, w, h) and normalize each image
    # to what is expected by resnet (a priori, works)
    obj_mean = [0.485, 0.456, 0.406]
    obj_std = [0.229, 0.224, 0.225]

    n_samples = image_batch.shape[0]

    for i in range(n_samples):
        for c in range(3):
            # First, scale to correct variance
            image_batch[i, c] = image_batch[i, c] * (obj_std[c] / image_batch[i, c].std())
            # Then, move the mean
            image_batch[i, c] = image_batch[i, c] + obj_mean[c] - image_batch[i, c].mean()
            # print(image_batch[i, c].shape, image_batch[i, c].mean().data.cpu().numpy(), obj_mean[c], image_batch[i, c].std().data.cpu().numpy(), obj_std[c])

        return image_batch

if __name__ == "__main__":

    # Create image with channel being the first dimension
    # Then swap for the last
    # =================================================
    one_channel_im = np.random.random((30,28))# not square to test if line and col are being kept in order
    rgb_im_first = to_rgb_channel_first(one_channel_im)
    assert rgb_im_first.shape[0] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_first.shape[0])
    assert rgb_im_first.shape[1] == 30, "n_line is not the same"

    specific_pixel_to_check = np.copy(rgb_im_first[:,20,18])
    rgb_im_last = channel_first_to_channel_last(rgb_im_first)
    specific_pixel_to_check_new = np.copy(rgb_im_last[20,18,:])
    assert rgb_im_last.shape[2] == 3, "wrong channel number \nis : {}, should be : 3".format(rgb_im_last.shape[2])
    assert rgb_im_last.shape[0] == 30, "n_line is not the same"
    assert np.all(specific_pixel_to_check == specific_pixel_to_check_new), "pixel changed"

    # Create image with channel being the last dimension
    # Then swap for the first
    # =================================================
    one_channel_im = np.random.random((30,28))
    rgb_im_last = to_rgb_channel_last(one_channel_im)
    assert rgb_im_last.shape[2] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_last.shape[2])
    assert rgb_im_last.shape[0] == 30, "n_line is not the same"

    specific_pixel_to_check = np.copy(rgb_im_last[20,18,:])
    rgb_im_first = channel_last_to_channel_first(rgb_im_last)
    specific_pixel_to_check_new = np.copy(rgb_im_first[:,20,18])
    assert rgb_im_first.shape[0] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_first.shape[0])
    assert rgb_im_first.shape[1] == 30, "n_line is not the same"
    assert np.all(specific_pixel_to_check == specific_pixel_to_check_new), "pixel changed"

    # Same tests, but for images in batches
    #=======================================

    # Create image with channel being the first dimension
    one_channel_im_batch = np.random.random((4, 30,28))# not square to test if line and col are being kept in order
    rgb_im_first = to_rgb_channel_first(one_channel_im_batch)
    assert rgb_im_first.shape[0] == 4, "wrong batch number \nis :{}, should be 4".format(rgb_im_first.shape[0])
    assert rgb_im_first.shape[1] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_first.shape[1])
    assert rgb_im_first.shape[2] == 30, "n_line is not the same"

    # Create image with channel being the last dimension
    # =================================================
    one_channel_im_batch = np.random.random((5, 30,28))
    rgb_im_last = to_rgb_channel_last(one_channel_im_batch)
    assert rgb_im_last.shape[0] == 5, "wrong batch number \nis :{}, should be 4".format(rgb_im_last.shape[0])
    assert rgb_im_last.shape[3] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_last.shape[3])
    assert rgb_im_last.shape[1] == 30, "n_line is not the same"

    print("Tests ok.")
