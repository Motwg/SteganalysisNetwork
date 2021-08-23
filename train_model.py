import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from steganalysis.network import Network
from steganalysis import config as c

try:
    epochs = int(input('Enter number of epochs [default 5]: ') or '5')
except ValueError:
    epochs = 5
dataset = input('Enter dataset filename [default "ds_comb_6890x4"]: ') or 'ds_comb_6890x4'
model = input('Enter catalog name to load model or leave blank to create new model: ') or ''
print('Starting')

set_size, batch_size = 1200, 16
steps_per_epoch = set_size // batch_size

try:
    network = Network(dataset)
    try:
        network.load_model(model)
    except OSError:
        print(f'Model "{model}" may not exists')
        print('Skipping loading model')
    network.labels_to_categorical()
    network.compile()
    network.fit(
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size
    )
    response = input('Enter name for your model or leave blank to abort: ')
    if response != '':
        network.model.save(c.MODELS_PATH + response, save_format='tf')
except FileNotFoundError:
    print(f'Dataset "{dataset}" may not exists')
    exit(-1)
