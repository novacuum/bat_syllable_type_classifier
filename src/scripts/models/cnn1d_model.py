import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

from engine.features.feature_extraction import FeaturesFileList
from engine.features.feature_sequence import FeatureSequence
from engine.files.lists import FileList
from engine.nn.training import NNModel
from scripts.models.datasets import Dataset

"""CNN model with 1D convolution for the experiments

Author: Gilles Waeber, VII 2019"""


def conv_pool_1d(dims):
    dims[0:1] -= 2
    dims[0:1] //= 2


def can_conv_pool_1d(dims):
    return np.all((dims[0:1] - 2) // 2 > 1)


def create_cnn_1d_model(input_shape, num_classes, learning_rate):
    # Create the model
    model = Sequential()
    dims = np.array(input_shape)

    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))  # Now is 256x128
    conv_pool_1d(dims)

    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))  # Now is 128x64
    conv_pool_1d(dims)

    if can_conv_pool_1d(dims):
        model.add(Conv1D(64, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))  # Now is 64x32
        conv_pool_1d(dims)

    if can_conv_pool_1d(dims):
        model.add(Conv1D(128, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))  # Now is 32x16
        conv_pool_1d(dims)

    if can_conv_pool_1d(dims):
        model.add(Conv1D(256, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))  # Now is 16x8
        conv_pool_1d(dims)

    if can_conv_pool_1d(dims):
        model.add(Conv1D(256, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))  # Now is 16x8
        conv_pool_1d(dims)

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['acc'])

    print(f'Model Summary (input shape: {model.input_shape})')
    model.summary()

    return model


def test_cnn_1d_model(
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
    else:
        prepare_args = prepare_args.copy()
    prepare_args['add_dim'] = False
    num_classes = len(dataset.classes)
    model_name = f'cnn_1d_{dataset.name}{model_suffix}'

    fs = FeatureSequence.from_htk(training.run().files[0].path())
    d1, d2 = fs.np_sequence.shape
    print(f'Sample shape: {fs.np_sequence.shape}')

    if 'padding_length' in prepare_args:
        d1 = prepare_args['padding_length']

    model = lambda: create_cnn_1d_model((d1, d2), num_classes, learning_rate)

    # Train the model
    m = training.create_nn_model(model_name, model, validate=validation, prepare_args=prepare_args)
    print(f'Model path: {m.train(max_epochs, save_every=save_every).files[0].path()}')
    k: NNModel = m.train(max_epochs, save_every=save_every).run()
    k.recognize_best_val_acc(testing=testing).extract(f'testing_{dataset.name}').run()
