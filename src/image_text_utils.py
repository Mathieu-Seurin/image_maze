import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from rl_agent.gpu_utils import FloatTensor, Tensor
from PIL import Image

import nltk
import string
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle as pkl

import random

class TokenizeStemSentence(object):

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.trans_table = str.maketrans(dict.fromkeys(string.punctuation))
        self.stop_word = nltk.corpus.stopwords.words('english')

    def tokenize_and_stem(self, sentence):
        sentence_word = []
        sentence_wo_punct = str.translate(sentence, self.trans_table)
        words = nltk.tokenize.word_tokenize(sentence_wo_punct)

        for raw_word in words:
            word = raw_word.lower()
            if word not in self.stop_word:
                sentence_word.append(self.stemmer.stem(word))

        return sentence_word

    def __call__(self, sentence):
        return self.tokenize_and_stem(sentence=sentence)


class TextToIds(object):
    def __init__(self):

        self.path_to_vocabulary = 'src/text/vocabulary.pkl'
        self.all_words = pkl.load(open(self.path_to_vocabulary, 'rb'))

        self.tokenizer = TokenizeStemSentence()
        self.word_to_id = LabelEncoder()
        all_words_encoded = self.word_to_id.fit_transform(np.array(list(self.all_words)))

        self.eos_id = float(self.word_to_id.transform(['eos'])[0])

        # self.id_to_one_hot = OneHotEncoder(sparse=False)
        # self.id_to_one_hot.fit(all_words_encoded.reshape(len(self.all_words),1))


    def sentence_to_matrix(self, sentence):
        sentence_array = np.array(self.tokenizer(sentence))
        sentence_encoded = self.word_to_id.transform(sentence_array)
        #sentence_one_hot = self.id_to_one_hot.transform(sentence_encoded.reshape(len(sentence_encoded), 1))
        #return sentence_one_hot

        return sentence_encoded

    def pad_batch_sentence(self, sentences):
        max_length = max(sentences, key=lambda x:x.size(1)).size(1)

        sentences_padded = [self.pad(sentence=sentence, max_length=max_length) for sentence in sentences]
        return sentences_padded

    def pad(self, sentence, max_length):
        n_padding = max_length-sentence.size(1)
        if n_padding != 0:
            padding = torch.ones(1,n_padding)*self.eos_id
            sentence = torch.cat((sentence, padding), dim=1)

        return sentence


class TextObjectiveGenerator(object):

    def __init__(self, zone_type, n_zone, sentence_file):

        assert n_zone <= 4, "At the moment, can only create 4 zones"
        self.path_to_text = 'src/text/'
        self.vocabulary_path = os.path.join(self.path_to_text, 'vocabulary.pkl')

        self.all_labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven','eight', 'nine',
                      'tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']

        if zone_type == 'zone':
            zone_name = ['cyan', 'ligthcyan', 'palegreen', 'lightpalegreen']
        elif zone_type == 'zone_color_gradient' :
            zone_name = ['downleft', 'downright', 'upleft', 'upright']
        elif zone_type == 'sequential': # NO ZONE
            zone_name = ['here']
        else:
            assert False, "Problem of grid type, can be zone or 'zone_color_gradient'"

        self.tokenize = TokenizeStemSentence()

        # Token like : begin of sentence, end of sentence etc ...
        self.special_tokens = ['eos']

        self.text2zone_str = zone_name
        self.zone_readable_used = zone_name[:len(self.text2zone_str)]

        self.all_sentences_template = []

        self.sentence_file = sentence_file
        self.load_sentences()

        # if os.path.exists(self.vocabulary_path):
        #     self.all_words = pkl.load(open(self.vocabulary_path, 'rb'))
        # else:
        self.build_vocabulary()


    def sample(self, label, zone):
        rand_sentence = random.choice(self.all_sentences_template)

        #convert label number to the actual name of the class
        object_str = self.all_labels[label]
        zone_readable = self.text2zone_str[zone]

        sentence_formatted = rand_sentence.format(object=object_str, color=zone_readable)
        return sentence_formatted

    def load_sentences(self):

        self.all_sentences_template = []
        with open(os.path.join(self.path_to_text, self.sentence_file)) as sentences_file:
            for sentence in sentences_file.readlines():
                sentence = sentence.replace('\n', '')
                self.all_sentences_template.append((sentence))

    def build_vocabulary(self):
        all_words = set([self.tokenize.stemmer.stem(word.lower()) for word in self.all_labels])

        sentence_word = []
        for sentence in self.all_sentences_template:
            sentence_replaced = sentence.replace('{color}', '')
            sentence_replaced = sentence_replaced.replace('{object}', '')

            sentence_word.extend(self.tokenize(sentence_replaced))

        sentence_word = set(sentence_word)
        self.all_words = all_words.union(sentence_word)
        self.all_words = self.all_words.union(self.zone_readable_used)

        self.all_words = self.all_words.union(set(self.special_tokens))

        #save vocabulary for later usage in model
        pkl.dump(self.all_words, open(self.vocabulary_path, 'wb'))

    @property
    def voc_size(self):
        return len(self.all_words)

def np_color_to_str(color):
    return '_'.join(map(str, map(int, color)))

def normalize_image_for_saving(img):
    # Use image normalization after loading?
    mean_per_channel = np.array([0.5450519, 0.88200397, 0.54505189])
    std_per_channel = np.array([0.35243599, 0.23492979, 0.33889725])

    assert img.shape == (3,28,28), "Problem with dimension"

    im = np.copy(img)

    im[0,:,:] = (im[0,:,:] - mean_per_channel[0]) / std_per_channel[0]
    im[1,:,:] = (im[1,:,:] - mean_per_channel[1]) / std_per_channel[1]
    im[2,:,:] = (im[2,:,:] - mean_per_channel[2]) / std_per_channel[2]
    return im

def load_image_or_fmap(path_to_images, last_chosen_file=None):

    if last_chosen_file:
        file_path = os.path.join(path_to_images, last_chosen_file+'.npy')
        img = np.load(file_path)
        return img, None

    all_files = [f for f in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, f))]
    chosen_file = np.random.choice(all_files)
    image_path = os.path.join(path_to_images, chosen_file)

    if 'raw' in path_to_images:
        img = Image.open(image_path)
        return img, chosen_file.split('.')[0]
    else:
        img = np.load(image_path)
    return img, None


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


def make_video(replay, filename, title=''):
    n_frames = len(replay)
    n_channels, n_w, n_h = replay[0].shape
    if not os.path.isdir(filename):
        os.makedirs(filename)
    for i in range(n_frames):
        plt.figure()
        plt.title('Objective : ' + title)
        plt.imshow((replay[i]/255).transpose(1, 2, 0))
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
