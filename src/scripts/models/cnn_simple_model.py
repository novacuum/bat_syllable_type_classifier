import numpy as np

from engine.features.feature_extraction import FeaturesFileList
from engine.features.feature_sequence import FeatureSequence
from engine.files.lists import FileList
from engine.nn.training import NNModel
from scripts.models.datasets import Dataset


"""CNN model experiments

Author: Gilles Waeber, VI 2019"""


def conv_pool_2d(dims):
    dims[0:2] -= 2
    dims[0:2] //= 2


def can_conv_pool_2d(dims):
    return np.all((dims[0:2] - 2) // 2 > 1)


def create_cnn_model(input_shape, num_classes, learning_rate):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import Adam

    # Create the model
    model = Sequential()
    dims = np.array(input_shape)

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_pool_2d(dims)  # Now is 256x128

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_pool_2d(dims)  # Now is 128x64

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['acc'])

    print(f'Model Summary (input shape: {model.input_shape})')
    model.summary()

    return model


def test_cnn_model(
    *,
    dataset: Dataset,
    learning_rate: float,
    max_epochs: int,
    training: FeaturesFileList,
    validation: FileList = None,
    testing: FileList = None,
    prepare_args: dict = None,
    save_every=100,
    model_suffix='',
):
    if prepare_args is None:
        prepare_args = {}
    num_classes = len(dataset.classes)
    model_name = f'cnn_simple_2d_{dataset.name}{model_suffix}'

    fs = FeatureSequence.from_htk(training.run().files[0].path())
    d1, d2 = fs.np_sequence.shape
    print(f'Sample shape: {fs.np_sequence.shape}')

    if 'padding_length' in prepare_args:
        d1 = prepare_args['padding_length']
    if 'new_dim_size' in prepare_args:
        d2 //= prepare_args['new_dim_size']
        d3 = prepare_args['new_dim_size']
    else:
        d3 = 1

    model = lambda: create_cnn_model((d1, d2, d3), num_classes, learning_rate)

    # Train the model
    m = training.create_nn_model(model_name, model, validate=validation, prepare_args=prepare_args)
    k: NNModel = m.train(max_epochs, save_every=save_every)
    print(f'Model path: {", ".join(f.path() for f in k.files)}')
    k.recognize_best_val_acc(testing=testing).extract(f'testing_{dataset.name}').run()
