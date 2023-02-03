from engine.features.feature_extraction import FeaturesFileList
from engine.features.feature_sequence import FeatureSequence
from engine.files.lists import FileList
from scripts.models.datasets import Dataset
from scripts.models.utils import train


def create_lstm_model(d2, num_classes, learning_rate, lstm_neurons=256):
    import tensorflow
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input, Embedding
    from tensorflow.keras.optimizers import Adam

    # Create the model
    model = Sequential()
    model.add(Input(shape=(None, d2), dtype=tensorflow.float32, ragged=True))

    # return_sequence: return the full sequence and not only the last element
    model.add(LSTM(lstm_neurons, return_sequences=True))
    model.add(LSTM(lstm_neurons, return_sequences=False))
    model.add(Dense(1024))
    model.add(Dropout(rate=.5))
    model.add(Dense(1024))
    model.add(Dropout(rate=.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['acc'])

    print(f'Model Summary (input shape: {model.input_shape})')
    model.summary()

    return model


def test_lstm_model(
        dataset: Dataset,
        learning_rate: float,
        max_epochs: int,
        training: FeaturesFileList,
        prepare_args: dict,
        validation: FileList = None,
        testing: FileList = None,
        save_every=1,
        model_suffix='',
        lstm_neurons=128,
):
    prepare_args = prepare_args.copy()
    prepare_args['add_dim'] = False

    num_classes = len(dataset.classes)
    model_name = f'lstm_{dataset.name}{model_suffix}'

    fs = FeatureSequence.from_htk(training.run().files[0].path())
    d1, d2 = fs.np_sequence.shape
    print(f'Sample shape: {fs.np_sequence.shape}')

    model = lambda: create_lstm_model(d2, num_classes, learning_rate, lstm_neurons)

    # Train the model
    train(model_name, dataset, model, max_epochs, prepare_args, training, save_every, validation, testing)
