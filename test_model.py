import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from steganalysis.config import MODELS_PATH
from steganalysis import dataset as ds
import numpy as np

model_name = input('Enter model catalog name [default "server_model"]: ') or 'server_model'
model = tf.keras.models.load_model(MODELS_PATH + model_name)
np.set_printoptions(suppress=True)

d = ds.load_dataset(f'ds_6890')
d.labels_to_categorical()
images = [np.asarray(img) for img in d.test_images[:, :, :, -1:]]
_, acc = model.evaluate(np.array(images), d.test_labels, verbose=2)
print('Cover - accuracy: {:5.2f}%'.format(100 * acc))

for alg in ['lsb', 'qim', 'dc_qim']:
    d = ds.load_dataset(f'ds_{alg}_6890')
    d.labels_to_categorical()
    images = [np.asarray(img) for img in d.test_images[:, :, :, -1:]]
    _, acc = model.evaluate(np.array(images), d.test_labels, verbose=2)
    print('Stego {} - accuracy: {:5.2f}%'.format(alg, 100 * acc))

d = ds.load_dataset('ds_comb_6890x4')
d.labels_to_categorical()
images = [np.asarray(img) for img in d.test_images[:, :, :, -1:]]
_, acc = model.evaluate(np.array(images), d.test_labels, verbose=2)
print('Combined dataset - accuracy: {:5.2f}%'.format(100 * acc))
