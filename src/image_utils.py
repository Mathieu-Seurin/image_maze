import numpy as np
from skvideo.io import FFmpegWriter as VideoWriter

def to_rgb_channel_first(im):
    w, h = im.shape
    ret = np.empty((3, w, h), dtype=np.float32)
    ret[0, :, :] = im / 255
    ret[1, :, :] = ret[2, :, :] = ret[0, :, :]
    return ret

def to_rgb_channel_last(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = im / 255
    ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
    return ret

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
    pass


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
    assert rgb_im_first.shape[0] == 3, f"wrong channel number \nis :{rgb_im_first.shape[0]}, should be 3"
    assert rgb_im_first.shape[1] == 30, "n_line is not the same"

    rgb_im_last = channel_first_to_channel_last(rgb_im_first)
    assert rgb_im_last.shape[2] == 3, f"wrong channel number \nis : {rgb_im_last.shape[2]}, should be : 3"
    assert rgb_im_last.shape[0] == 30, "n_line is not the same"

    # Create image with channel being the last dimension
    # Then swap for the first
    # =================================================
    one_channel_im = np.random.random((30,28))
    rgb_im_last = to_rgb_channel_last(one_channel_im)
    assert rgb_im_last.shape[2] == 3, f"wrong channel number \nis :{rgb_im_last.shape[2]}, should be 3"
    assert rgb_im_last.shape[0] == 30, "n_line is not the same"

    rgb_im_first = channel_last_to_channel_first(rgb_im_last)
    assert rgb_im_first.shape[0] == 3, f"wrong channel number \nis :{rgb_im_first.shape[0]}, should be 3"
    assert rgb_im_first.shape[1] == 30, "n_line is not the same"

    print("Tests ok.")
