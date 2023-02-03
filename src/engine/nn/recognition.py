from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from utils.file import to_local_data_path, expand_data_path, to_unix_path
from .properties import ModelProperties
from ..files.files import FileType, File
from ..files.tasks import TransformationTask
from ..helpers import write_file
from ..metadata import metadata_db
from ..nn.reporting import ResultsJSONFileList
from ..recognition import Result
from ..utils import gpu_free
from ..utils import mkdir

if TYPE_CHECKING:
    from engine.features.feature_extraction import FeaturesFileList


def recognize_nn(model_path, ffl: FeaturesFileList, dest_file, model_props: ModelProperties):
    """Recognize the samples in batch"""
    model: Model = load_model(model_path)
    # assert len(model.loss_functions) == 1, "Not implemented"
    assert len(model.compiled_loss._losses) == 1, "Not implemented"

    files = ffl.files

    x, y, _ = ffl.get_xyw(model_props)

    y_truth = [f.metadata.label for f in files]
    y_pred = model.predict(x)
    y_pred_text = [model_props.num_label[z] for z in np.argmax(y_pred, axis=1)]
    #@simon change reduction to none! we want individual losses
    #https://www.tensorflow.org/api_docs/python/tf/keras/losses/Reduction
    # model.loss_functions[0].reduction = 'none'
    model.compiled_loss._losses[0].reduction = 'none'
    # losses = K.eval(model.loss_functions[0](y_true=K.constant(y), y_pred=K.constant(y_pred)))
    losses = K.eval(model.compiled_loss._losses[0](y_true=K.constant(y), y_pred=K.constant(y_pred)))

    results = [Result(f.path(), t, p, l) for f, t, p, l in zip(files, y_truth, y_pred_text, losses)]

    write_file(dest_file, json.dumps([r.serialize() for r in results], indent=2))

    gpu_free()


class RecognizeNNTask(TransformationTask):
    validate: FeaturesFileList

    def __init__(self, src_list, validate, *, model_path=None, epoch=None):
        assert (model_path is None) == (epoch is None)
        if validate.type == FileType.AUDIO:
            validate = validate.preproc_from(src_list)
        assert validate.type == FileType.FEATURES
        assert len(src_list.files) == 1
        if model_path is None:
            model = src_list.files[0]
            epoch = src_list.current_epoch
        else:
            model = File(expand_data_path(model_path))
        self.validate = validate
        self.model_props = src_list.model_props

        super().__init__(src_list, ResultsJSONFileList([
            File(f"{model.folder}/rec",
                 f"rec_{validate.features_digest()}_e{epoch}.json", model,
                 metadata=metadata_db(validate.task.src_list.files))
        ], self), {'validate': validate.task.export(), 'model_path': to_unix_path(to_local_data_path(model_path))})

    def run(self, missing, *, parallel=None):
        self.validate.run(parallel=parallel)
        super().run(missing)

    def run_file(self, file):
        mkdir(file.folder)
        recognize_nn(file.task_src.path(), self.validate, file.path(), self.model_props)
