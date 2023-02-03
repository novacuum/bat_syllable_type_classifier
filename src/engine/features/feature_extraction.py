from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from os import path
from typing import TYPE_CHECKING, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from engine.settings import BIRDVOICE_DATA_DIR
from engine.features.feature_sequence import FeatureSequence, merge_to_x
from engine.files.files import File, FileType
from engine.files.lists import FileList
from engine.files.tasks import PreprocessingTask, VirtualTransformationTask
from engine.helpers import write_lines
from engine.hmm.training import CreateHMMModelTask, BlankModelFileList
from engine.settings import BIN_HWRECOG
from engine.utils import mkdir, print_log, list_ellipsis, call
from utils.file import to_unix_path, to_local_data_path

if TYPE_CHECKING:
    from engine.nn.properties import ModelProperties
    from engine.nn.training import NNModel
    from engine.files.parallel import ParallelFileList

"""Feature Extraction

Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018
Extract features from black and white pictures into htk files

Usage:
  f = ...  # A feature files collection, Geo or HoG (see spectrograms.py)
  bm = ...  # A blank model (see training.py)
  bm = f.create_model(name, number_of_states) # Create a model from feature files
"""


def extract_geo_features(src_file, dest_file, x_per_sec):
    from engine.features.geometric import GeoFeatures
    geo = GeoFeatures(src_file)
    geo.extract_features()
    geo.feature_sequence.x_per_sec = x_per_sec
    geo.feature_sequence.to_htk(dest_file)


def extract_hog_hwr_features(src_folder, src_files, dest_folder):
    """Extract HOG features using the HWRecog software"""
    file_ids = f"{src_folder}/hog_ids.tmp"
    src_ids = [f for f in src_files]
    print_log(f"  Write temp ids to {file_ids}: {list_ellipsis(src_ids)}")
    write_lines(file_ids, src_ids)
    call([
        'java', '-jar', BIN_HWRECOG,
        '-p', src_folder,
        '-h', dest_folder,  # Extract HoG
        '-i', file_ids,
        '-np'  # No preprocessing
    ], print_stdout=False)


def extract_geo_hwr_features(src_folder, src_files, dest_folder, progress=None):
    """Extract geometric features using the HWRecog software"""
    file_ids = f"{src_folder}/geo_hwrecog_ids.tmp"
    src_ids = [f for f in src_files]
    print_log(f"  Write temp ids to {file_ids}: {list_ellipsis(src_ids)}")
    write_lines(file_ids, src_ids)
    progress.total -= len(src_ids) - 1
    call([
        'java', '-jar', BIN_HWRECOG,
        '-p', src_folder,
        '-g', dest_folder,  # Extract Geo
        '-i', file_ids,
        '-np'  # No preprocessing
    ], progress=progress, print_stdout=False)


class ExtractGeoFeaturesTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, FeaturesFileList([
            File(f.folder, f"{f.name}.htk", f) for f in src_list.files
        ], self), {})
        from engine.spectrograms import get_x_per_sec
        self.x_per_sec = get_x_per_sec(src_list)

    def __str__(self):
        return 'Extract geometric features'

    def run_file(self, file):
        mkdir(file.folder)
        extract_geo_features(file.task_src.path(), file.path(), self.x_per_sec)


class ExtractHOGFeaturesTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, FeaturesFileList([
            File(f"{f.folder}/hog2", f"{f.p.stem}.htk", f)
            for f in src_list.files
        ], self), {})
        from engine.spectrograms import get_x_per_sec
        self.x_per_sec = get_x_per_sec(src_list)

    def __str__(self):
        return 'Extract HOG features'

    def run_file(self, file: File):
        from engine.features.hog import extract_hog_features
        mkdir(file.folder)
        extract_hog_features(file.task_src.path(), file.path(), self.x_per_sec)


# noinspection PyAbstractClass
class ExtractHOGHWRecogTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, FeaturesFileList([
            File(f"{f.folder}/hog", f"{f.p.stem}.htk", f)
            for f in src_list.files
        ], self), {})

    def run(self, missing, *, parallel=None):
        self.src_list.run(parallel=parallel)
        print_log('  Extract HoG features (HWRecog)')
        by_folder = defaultdict(set)

        for file in missing:
            by_folder[file.folder].add(file)

        for folder, files in by_folder.items():
            src_files = []
            src_folder = next(iter(files)).task_src.folder
            for file in files:
                if path.exists(file.path()):
                    print_log(f"  {file.path()} already exists")
                else:
                    src_files.append(file.task_src.p.stem)
            if len(src_files) > 0:
                mkdir(folder)
                extract_hog_hwr_features(src_folder, src_files, folder)


# noinspection PyAbstractClass
class ExtractGeoHWRecogTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, FeaturesFileList([
            File(f"{f.folder}/geo_hwr", f"{f.p.stem}.htk", f)
            for f in src_list.files
        ], self), {})

    def run(self, missing, *, parallel=None):
        self.src_list.run(parallel=parallel)
        print_log('  Extract geometric features (HWRecog)')
        by_folder = defaultdict(list)

        for file in missing:
            by_folder[file.folder].append(file)

        for folder, files in by_folder.items():
            src_files = []
            src_folder = files[0].task_src.folder
            for file in files:
                if path.exists(file.path()):
                    print_log(f"  {file.path()} already exists")
                else:
                    src_files.append(file.task_src.p.stem)
            if len(src_files) > 0:
                mkdir(folder)
                extract_hog_hwr_features(src_folder, src_files, folder)


def extract_pixel_features(src_path, dest_path, x_per_sec):
    data: np.ndarray = plt.imread(src_path)

    # Keep first channel only and transpose
    data = data[:, :, 0].transpose()

    FeatureSequence(data, x_per_sec).to_htk(dest_path)


class ExtractPixelsTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, FeaturesFileList([
            File(f'{f.folder}/img', f'{f.p.stem}.htk', f)
            for f in src_list.files
        ], self), {})
        from engine.spectrograms import get_x_per_sec
        self.x_per_sec = get_x_per_sec(src_list)

    def run_file(self, file):
        mkdir(file.folder)
        extract_pixel_features(file.task_src.path(), file.path(), self.x_per_sec)


# noinspection PyAbstractClass
class RepeatFeaturesTask(VirtualTransformationTask):
    def __init__(self, src_list: FileList):
        by_label = src_list.by_label()
        most_label = len(max(by_label.values(), key=len))
        for l in by_label.keys():
            if len(by_label[l]) * 2 <= most_label:
                by_label[l] = by_label[l] * (most_label // len(by_label[l]))
            if len(by_label[l]) < most_label:
                by_label[l].extend(by_label[l][:most_label - len(by_label[l])])
        files = [f for l in by_label.values() for f in l]
        super().__init__(src_list, src_list.__class__(
            files=files,
            task=self
        ), {})


class FeaturesFileList(FileList):
    type = FileType.FEATURES
    __slots__ = ('_xyw', '_variable_length', '_model_props')
    _xyw: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], None]
    _variable_length: bool
    _model_props: ModelProperties

    def __init__(self, files, task=None):
        super().__init__(files, task)
        self._xyw = None

    def preproc_from(self, from_list) -> FeaturesFileList:
        """
        If preproc_from is called on the feature list, do nothing
        This method is for keeping the same api between FeaturesFileList and AudioFileList
        """
        return self

    def multi(self) -> FeaturesFileList:
        """Repeat the under-represented classes"""
        task = RepeatFeaturesTask(self)
        return task.dest_list

    def k_fold(self, k, val_bins=1, test_bins=1) -> Union[ParallelFileList, FeaturesFileList]:
        """Use transparent K-Fold

        After a call to this method, all other actions are realized in parallel over all folds.
        The validation and the testing dataset are automatically injected.
        See the KFoldParallelList wrapper for more information."""
        from engine.k_fold import KFoldSeparationTask
        task = KFoldSeparationTask(self, k, val_bins=val_bins, test_bins=test_bins)
        return task.dest_list

    def create_hmm_model(self, name, states) -> BlankModelFileList:
        task = CreateHMMModelTask(self, name, states)
        return task.dest_list

    def create_nn_model(self, name, model, *, validate=None, prepare_args=None, fit_args=None) -> NNModel:
        from engine.nn.training import CreateNNModelTask
        import tensorflow as tf
        with tf.profiler.experimental.Trace("CreateModel"):
            task = CreateNNModelTask(self, name, model, validate=validate, prepare_args=prepare_args, fit_args=fit_args)
        return task.dest_list

    def files_digest_string(self):
        """String that is used to compute the list hash"""
        return '\n'.join(to_unix_path(to_local_data_path(f.p)) for f in self.files)

    def features_digest(self):
        """Create a name for a model, based on a hash of the source files paths"""
        digest = sha1(self.files_digest_string().encode()).hexdigest()
        return f"{digest}"

    def _lazy_load(self, model_props: ModelProperties):
        if self._xyw is not None and self._model_props == model_props:
            return

        x = merge_to_x([FeatureSequence.from_htk(f.path()) for f in self.files],
                       **model_props.prepare_args)
        y_text = [f.metadata.label for f in self.files]
        y = np.concatenate([model_props.label_cat[l] for l in y_text], axis=0)
        weights = np.array([model_props.label_weight[l] for l in y_text])
        # print(f'Weights: {model_props.label_weight}')

        self._model_props = model_props
        self._xyw = (x, y, weights)
        self._variable_length = model_props.prepare_args['variable_length'] if 'variable_length' in model_props.prepare_args else False

    def get_xyw(self, model_props: ModelProperties = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if model_props is None:
            from engine.nn.properties import ModelProperties
            model_props = ModelProperties(self)
        self._lazy_load(model_props)
        return self._xyw
