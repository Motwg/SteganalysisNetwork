import pickle
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
from PIL import Image

from .config import DATASETS_PATH
from .renamed_pickler import renamed_load


def save_dataset(dataset, filename):
    assert isinstance(dataset, Dataset)
    with open(f'{DATASETS_PATH}{filename}.pickle', 'wb') as outfile:
        pickle.dump(dataset, outfile)


def load_dataset(filename):
    with open(f'{DATASETS_PATH}{filename}.pickle', 'rb') as infile:
        return renamed_load(infile)


class Dataset:

    def __init__(self, train, test):
        assert isinstance(train, tuple) or isinstance(train, list)
        assert isinstance(test, tuple) or isinstance(test, list)
        assert len(train) == 2 and len(test) == 2

        self.train_images = train[0]
        self.train_labels = train[1]
        self.test_images = test[0]
        self.test_labels = test[1]

    def __iadd__(self, other):
        assert isinstance(other, Dataset)
        self.train_images = np.append(self.train_images, other.train_images, 0)
        self.train_labels = np.append(self.train_labels, other.train_labels, 0)
        self.test_images = np.append(self.test_images, other.test_images, 0)
        self.test_labels = np.append(self.test_labels, other.test_labels, 0)
        return self

    def save(self, filename):
        with open(f'{DATASETS_PATH}{filename}.pickle', 'wb') as outfile:
            pickle.dump(self, outfile)

    def save_cropped(self, filename):
        self.train_images, self.test_images = self._crop()
        self.train_images = np.asarray(self.train_images)
        self.test_images = np.asarray(self.test_images)
        self.save(filename)

    def save_resized(self, filename):
        self.train_images, self.test_images = self._resize()
        self.train_images = np.asarray(self.train_images)
        self.test_images = np.asarray(self.test_images)
        self.save(filename)

    def _crop(self):
        def crop(img):
            return np.array(Image.fromarray(img).crop((0, 0, 256, 256)))

        return [crop(img) for img in self.train_images], [crop(img) for img in self.test_images]

    def _resize(self):
        def resize(img):
            return np.array(Image.fromarray(img).resize((256, 256)))

        return [resize(img) for img in self.train_images], [resize(img) for img in self.test_images]

    # get data in (x, y), (z, t) format
    def get_data(self):
        return (self.train_images,
                self.train_labels), \
               (self.test_images,
                self.test_labels)

    def get_channel_data(self, channel):
        return (self.train_images[:, :, :, channel],
                self.train_labels), \
               (self.test_images[:, :, :, channel],
                self.test_labels)

    def get_size(self):
        return len(self.train_images), len(self.test_images)

    def shuffle(self):
        self.train_images, self.train_labels = (shuffle(self.train_images, self.train_labels))
        # self.test_images, self.test_labels = (shuffle(self.test_images, self.test_labels))

    def labels_to_categorical(self, no_classes=2):
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_classes=no_classes)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes=no_classes)
