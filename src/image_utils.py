import numpy as np
from skvideo.io import FFmpegWriter as VideoWriter
import matplotlib.pyplot as plt

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
    writer = VideoWriter(filename + '.mp4')
    for i in range(n_frames):
        writer.writeFrame(replay[i]*255)
    writer.close()


if __name__ == "__main__":

    # Create image with channel being the first dimension
    # Then swap for the last
    # =================================================
    one_channel_im = np.random.random((30,28))# not square to test if line and col are being kept in order
    rgb_im_first = to_rgb_channel_first(one_channel_im)
    assert rgb_im_first.shape[0] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_first.shape[0])
    assert rgb_im_first.shape[1] == 30, "n_line is not the same"

    rgb_im_last = channel_first_to_channel_last(rgb_im_first)
    assert rgb_im_last.shape[2] == 3, "wrong channel number \nis : {}, should be : 3".format(rgb_im_last.shape[2])
    assert rgb_im_last.shape[0] == 30, "n_line is not the same"

    # Create image with channel being the last dimension
    # Then swap for the first
    # =================================================
    one_channel_im = np.random.random((30,28))
    rgb_im_last = to_rgb_channel_last(one_channel_im)
    assert rgb_im_last.shape[2] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_last.shape[2])
    assert rgb_im_last.shape[0] == 30, "n_line is not the same"

    rgb_im_first = channel_last_to_channel_first(rgb_im_last)
    assert rgb_im_first.shape[0] == 3, "wrong channel number \nis :{}, should be 3".format(rgb_im_first.shape[0])
    assert rgb_im_first.shape[1] == 30, "n_line is not the same"


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
