from tensorflow import keras
from keras.layers.core import Dense, Flatten
from keras.layers import Input, Concatenate
from .dataset import load_dataset
from .config import MODELS_PATH


def get_vgg_model():
    img_input = Input(shape=(256, 256, 1))
    img_conc = Concatenate()([img_input] * 3)
    model = keras.Sequential()
    vgg = keras.applications.VGG19(
        include_top=False,
        weights='imagenet',
        input_tensor=img_conc,
        input_shape=(256, 256, 3),
    )
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(16, activation='relu', name='dense_1'))
    model.add(Dense(2, activation='softmax', name='dense_0'))

    return model


class Network:

    def __init__(self, dataset_filename):
        self.model = get_vgg_model()
        assert isinstance(self.model, keras.Sequential)

        d = load_dataset(dataset_filename)
        d.shuffle()
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = d.get_data()
        self.train_images = self.train_images[:, :, :, -1:]
        self.test_images = self.test_images[:, :, :, -1:]
        # print(f'Train Image shape: {self.train_images.shape}')
        # print(f'Train Label shape: {self.train_labels.shape}')
        # print(f'Test  Image shape: {self.test_images.shape}')
        # print(f'Test  Label shape: {self.test_labels.shape}')
        # self.model.summary()

    def load_model(self, catalog_name):
        self.model = keras.models.load_model(MODELS_PATH + catalog_name)

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-6),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def fit(self,
            batch_size=None,
            epochs=1,
            validation_freq=1,
            steps_per_epoch=1,
            callbacks=None):
        return self.model.fit(
            self.train_images,
            self.train_labels,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=(self.test_images[:], self.test_labels[:]),
            validation_freq=validation_freq,
            callbacks=callbacks
        )

    def labels_to_categorical(self):
        self.train_labels = keras.utils.to_categorical(self.train_labels, num_classes=2)
        self.test_labels = keras.utils.to_categorical(self.test_labels, num_classes=2)

    def summary(self):
        self.model.summary()
