from __future__ import annotations

import gc
from collections import defaultdict
from typing import Mapping, Tuple, Optional, MutableMapping, TYPE_CHECKING, Sequence

import numpy as np

from ..files.files import File, FileType

if TYPE_CHECKING:
    from engine.features.feature_extraction import FeaturesFileList


def count_by_label(files: Sequence[File]):
    r = defaultdict(lambda: 0)
    for e in files:
        r[e.metadata.label] += 1
    return dict(r)


class ModelProperties:
    __slots__ = (
        'feature_files', 'label_num', 'label_cat', 'label_weight', 'num_label', 'validate', 'prepare_args', 'fit_args',
        '_xyw')
    feature_files: FeaturesFileList
    label_num: Mapping[str, int]
    label_cat: Mapping[str, np.ndarray]
    label_weight: Mapping[str, float]
    num_label: Mapping[int, str]
    validate: Optional[FeaturesFileList]
    prepare_args: Mapping
    fit_args: MutableMapping
    _xyw: Tuple[np.ndarray, np.ndarray, np.ndarray]

    def __init__(self, feature_files: FeaturesFileList, *, validate=None, prepare_args=None, fit_args=None):
        from tensorflow.keras.utils import to_categorical

        if prepare_args is None:
            prepare_args = {}
        if fit_args is None:
            fit_args = {}
        if validate is not None:
            if validate.type == FileType.AUDIO:
                validate = validate.preproc_from(feature_files)
            assert validate.type == FileType.FEATURES
            self.validate = validate
        else:
            self.validate = None
        self._xyw = None
        self.prepare_args = prepare_args
        self.fit_args = fit_args
        self.feature_files = feature_files
        labels = sorted(set(f.metadata.label for f in feature_files.files))
        label_count = count_by_label(feature_files.files)
        self.label_weight = dict((l, len(feature_files.files) / len(labels) / c) for l, c in label_count.items())
        self.label_num = dict((v, k) for k, v in enumerate(labels))
        self.label_cat = dict((v, to_categorical(np.array([[k]]), len(labels))) for k, v in enumerate(labels))
        self.num_label = dict((k, v) for k, v in enumerate(labels))

    def decode_predictions(self, pred):
        return [
            sorted([(k, self.num_label[k], v) for k, v in enumerate(p)], key=lambda i: i[2], reverse=True)
            for p in pred
        ]

    def get_features_xyw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._xyw is None:
            self._xyw = self.feature_files.get_xyw(self)
            if self.validate is not None:
                self.fit_args['validation_data'] = self.validate.get_xyw(self)
        return self._xyw

    def free_memory(self):
        self._xyw = None
        del self.fit_args['validation_data']
        self.feature_files._xyw = None
        gc.collect()
