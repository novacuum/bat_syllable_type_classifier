from engine.features.feature_extraction import FeaturesFileList
from engine.features.feature_sequence import FeatureSequence
from engine.files.lists import FileList
from scripts.models.datasets import Dataset
from scripts.models.utils import train


# Review: DenseNet — Dense Convolutional Network (Image Classification)
# https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803

def create_densnet_model(input_shape, num_classes, learning_rate):
    from tensorflow.keras import applications, Model
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import Adam

    densnet_model = applications.DenseNet121(
        include_top=False,
        weights=None,
        input_tensor=None,
        input_shape=input_shape,
        pooling=None,
        classes=num_classes,
    )

    # Transfer learning using Keras with DenseNet-169
    # https://medium.com/@carlosz22/transfer-learning-using-keras-with-densenet-169-91679300f94a
    # https://gist.githubusercontent.com/carlosz22/70d98179cd70a17b7d1f1a0707e81d83/raw/4b1016ac9b2439dbd3fcea21c1b964dac43dbd04/transfer-learning-densenet-cifar.py
    # initializer = K.initializers.he_normal(seed=32)
    # layer = K.layers.Flatten()(layer)
    # layer = K.layers.BatchNormalization()(layer)
    # layer = Dense(units=256,activation='relu', kernel_initializer=initializer)(layer)
    # layer = K.layers.Dropout(0.4)(layer)
    # layer = K.layers.BatchNormalization()(layer)
    # layer = Dense(units=128,activation='relu', kernel_initializer=initializer)(layer)
    # layer = Dropout(0.4)(layer)
    # layer = Dense(units=10,activation='softmax', kernel_initializer=initializer)(layer)


    x = densnet_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(rate=.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=densnet_model.input, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['acc'])

    print(f'Model Summary (input shape: {model.input_shape})')
    model.summary()

    return model


def test_densnet_model(
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
    model_name = f'densNet_{dataset.name}{model_suffix}'

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

    model = lambda: create_densnet_model((d1, d2, d3), num_classes, learning_rate)

    # Train the model
    train(model_name, dataset, model, max_epochs, prepare_args, training, save_every, validation, testing)
