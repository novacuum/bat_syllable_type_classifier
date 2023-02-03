from engine.features.feature_extraction import FeaturesFileList
from engine.files.lists import FileList
from engine.nn.training import NNModel
from scripts.models.datasets import Dataset


def train(
        model_name
        , dataset: Dataset
        , model
        , max_epochs: int
        , prepare_args: dict
        , training: FeaturesFileList
        , save_every=100
        , validation: FileList = None
        , testing: FileList = None
):
    # Train the model
    m = training.create_nn_model(model_name, model, validate=validation, prepare_args=prepare_args)
    k: NNModel = m.train(max_epochs, save_every=save_every)
    print(f'Model path: {", ".join(f.path() for f in k.files)}')
    k.recognize_best_val_acc(testing=testing).extract(f'testing_{dataset.name}').run()
