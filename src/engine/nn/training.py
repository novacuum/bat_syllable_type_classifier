from __future__ import annotations

import json, pandas
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

from tensorflow.python.distribute.mirrored_strategy import MirroredStrategy
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import load_model
from tqdm import tqdm

from .prediction import PredictNNTask, PredictionResultList
from .properties import ModelProperties
from ..files.files import File, FileType
from ..files.lists import FileList
from ..files.tasks import TransformationTask, CreateModelTask
from ..helpers import read_file, write_file, TmpFile
from ..nn.recognition import RecognizeNNTask
from ..nn.reporting import ResultsJSONFileList
from ..settings import MODELS_FOLDER
from ..utils import print_log, mkdir, gpu_free


class NNTrainCallback(Callback):

    def __init__(self, model_trainer: NNModelTrainer, progress):
        super().__init__()
        self.mt = model_trainer
        self.progress = progress
        self.stats = json.loads(read_file(self.mt.stats_path(self.mt.trained_epoch)))
        self.validate = model_trainer.model_props.validate is not None

    def on_epoch_end(self, keras_epoch, logs=None):
        epoch = self.mt.trained_epoch + keras_epoch + 1
        acc, loss = logs.get('acc'), logs.get('loss')
        stat = {'epoch': epoch, 'acc': acc, 'loss': loss}
        line = f'Epoch {epoch} - train loss: {loss:.4f} - train acc: {acc:.2%}'
        if self.validate:
            val_acc, val_loss = logs.get('val_acc'), logs.get('val_loss')
            stat['val_acc'], stat['val_loss'] = val_acc, val_loss
            line += f' - val loss: {val_loss:.4f} - val acc: {val_acc:.2%}'
        if self.mt.test_better_train_loss(epoch, loss):
            line += ' (new best train loss)'
        # noinspection PyUnboundLocalVariable
        if self.validate and self.mt.test_better_val_acc(epoch, val_acc):
            line += ' (new best val acc)'
        tqdm.write(line)
        self.stats.append(stat)
        self.progress.update()

        # We save every epoch
        # But then we remove the ones that we don't intend on keeping
        # write_file(self.mt.stats_path(epoch), json.dumps(self.stats))
        write_file(self.mt.stats_path(epoch), pandas.Series(self.stats).to_json(orient='values'))
        with TmpFile(self.mt.model_path(epoch)) as out:
            self.model.save(out, save_format='h5')
        prev_epoch = epoch - 1
        if prev_epoch >= self.mt.src_model.current_epoch and prev_epoch % self.mt.save_every != 0:
            prev_stats, prev_model = self.mt.stats_path(prev_epoch), self.mt.model_path(prev_epoch)
            if prev_stats.is_file():
                os.remove(prev_stats)
            if prev_model.is_file():
                os.remove(prev_model)


@dataclass(frozen=True)
class ModelPath:
    path: str
    epoch: int


def parse_model_name(path):
    match = re.search(r"epoch_(?P<epoch>\d+).h5$", path)
    if match is None:
        return None
    return ModelPath(path, int(match.group('epoch')))


@dataclass(frozen=True)
class BestModelPath:
    path: str
    epoch: int
    value: float


def parse_best_model_name(path: Path):
    match = re.search(r"best_[a-z0-9_]+_e(?P<epoch>\d+)_v(?P<value>[\de.+-]+).h5$", path.name)
    if match is None:
        # print(f'No match with {path}')
        return None
    # print(f'Path {path} gave epoch {int(match.group("epoch"))}, value {float(match.group("value"))}')
    return BestModelPath(path, int(match.group('epoch')), float(match.group('value')))


@dataclass()
class BestModels:
    val_acc: Optional[BestModelPath]
    train_loss: Optional[BestModelPath]


def find_best_models(folder):
    """Find the best model explicitly named as such"""
    folder = Path(folder)
    best_val_acc = [c for c in (parse_best_model_name(f) for f in folder.glob('best_val_acc_e*_v*.h5')) if
                    c is not None]
    best_val_acc = sorted(best_val_acc, reverse=True, key=lambda c: c.value)
    best_train_loss = [c for c in
                       (parse_best_model_name(f) for f in folder.glob('best_train_loss_e*_v*.h5')) if
                       c is not None]
    best_train_loss = sorted(best_train_loss, reverse=False, key=lambda c: c.value)

    return BestModels(
        best_val_acc[0] if len(best_val_acc) else None,
        best_train_loss[0] if len(best_train_loss) else None,
    )


class NNModelTrainer:
    model_props: ModelProperties

    def __init__(self, folder, src_model: NNModel, epochs, *, save_every, validate=None):
        self.folder = folder
        self.src_model = src_model
        self.final_epoch = src_model.current_epoch + epochs
        self.save_every = save_every
        self.trained_epoch = self.find_latest_epoch()
        self.num_epochs = self.final_epoch - self.trained_epoch
        self.model_props = src_model.model_props
        self.validate = validate
        self.best = find_best_models(self.folder)

        mirrored_strategy = MirroredStrategy()
        with mirrored_strategy.scope():
            self.model: Model = load_model(str(self.model_path(self.trained_epoch)))

    def best_model_path(self, metric, epoch, value):
        return Path(f'{self.folder}/best_{metric}_e{epoch:03d}_v{value:.4g}.h5')

    def test_better_train_loss(self, epoch, loss):
        if self.best.train_loss is None or loss < self.best.train_loss.value:
            path = self.best_model_path('train_loss', epoch, loss)
            with TmpFile(path) as out:
                self.model.save(out, save_format='h5')
            if self.best.train_loss is not None:
                os.remove(self.best.train_loss.path)
            self.best.train_loss = parse_best_model_name(path)
            return True
        return False

    def test_better_val_acc(self, epoch, val_acc):
        if self.best.val_acc is None or val_acc > self.best.val_acc.value + .0001:
            path = self.best_model_path('val_acc', epoch, val_acc)
            with TmpFile(path) as out:
                self.model.save(out, save_format='h5')
            if self.best.val_acc is not None:
                os.remove(self.best.val_acc.path)
            self.best.val_acc = parse_best_model_name(path)
            return True
        return False

    def model_path(self, epoch):
        return Path(f'{self.folder}/epoch_{epoch:03d}.h5')

    def stats_path(self, epoch):
        return Path(f'{self.folder}/epoch_{epoch:03d}.h5.stats.json')

    def find_latest_epoch(self):
        all_models = [parse_model_name(f.name) for f in Path(self.folder).glob('epoch_*.h5')]
        all_models = [c for c in all_models
                      if c is not None and self.src_model.current_epoch < c.epoch <= self.final_epoch]
        all_models = sorted(all_models, reverse=True, key=lambda m: m.epoch)
        return all_models[0].epoch if len(all_models) else self.src_model.current_epoch

    def train(self):
        x, y, weights = self.model_props.get_features_xyw()

        print_log(f'Training from epoch {self.trained_epoch} to epoch {self.final_epoch}')

        with tqdm(initial=self.trained_epoch, total=self.final_epoch, dynamic_ncols=True,
                  desc=self.src_model.task.get_model_task().name) as progress_all:
            nntc = NNTrainCallback(self, progress_all)

            # Default batch_size is 32
            self.model.fit(x, y, verbose=0, sample_weight=weights, epochs=self.num_epochs, callbacks=[nntc], **self.model_props.fit_args)
            write_file(self.stats_path(self.final_epoch), pandas.Series(nntc.stats).to_json(orient='values'))

        with TmpFile(self.model_path(self.final_epoch)) as out:
            self.model.save(out, save_format='h5')

        return self

    def clean(self):
        gpu_free()
        self.src_model.model_props.free_memory()


class CreateNNModelTask(CreateModelTask):
    def __init__(self, src_list, name, model: Union[Model, dict], *, validate=None, prepare_args=None, fit_args=None):
        self.folder = MODELS_FOLDER / f'nn_{name}' / src_list.features_digest()
        self.config_path = self.folder / 'model_config.json'
        self.model_path = self.folder / 'epoch_000.h5'
        self.src_list = src_list

        # @todo change to tf save_model format -> model_path becomes a directory!
        if self.model_path.is_file() and self.config_path.is_file():
            model = None
            config = json.loads(read_file(self.config_path))
        elif self.model_path.is_file():
            model = None
            config = load_model(str(self.model_path)).get_config()
            write_file(self.config_path, json.dumps(config))
        elif isinstance(model, dict):
            config = model
        elif isinstance(model, Model):
            # In this case, we write the model right away to prevent issues if the memory is cleared
            self.src_list.run()
            config = model.get_config()
            self.write_model(model)
            model = None
        elif callable(model):
            # In this case, we write the model right away to prevent issues if the memory is cleared
            self.src_list.run()
            model = model()
            config = model.get_config()
            self.write_model(model)
            model = None
        else:
            raise ValueError('Invalid config')

        self.mp = ModelProperties(src_list, validate=validate, prepare_args=prepare_args, fit_args=fit_args)
        super().__init__(src_list, NNModel([
            File(self.model_path, task_src=src_list.files)
        ], self, self.mp), {'name': name, 'model': config, 'prepare_args': prepare_args, 'fit_args': fit_args}, name)
        self.model = model

    def run(self, missing, *, parallel=None):
        if self.mp.validate is not None:
            self.mp.validate.run(parallel=parallel)
        super().run(missing, parallel=parallel)

    def run_file(self, file):
        print(file.path())
        assert self.model is not None
        model = self.model if isinstance(self.model, Model) else Model.from_config(self.model)
        self.write_model(model)

    def write_model(self, model):
        mkdir(self.folder)
        write_file(f'{self.model_path}.stats.json', '[]')
        write_file(self.folder / 'files.txt', self.src_list.files_digest_string())
        write_file(self.config_path, json.dumps(model.get_config()))
        with TmpFile(self.model_path) as out:
            model.save(out, save_format='h5')
        gpu_free()


class TrainNNModelTask(TransformationTask):
    def __init__(self, src_list: NNModel, epochs, save_every):
        props = {'epochs': epochs, 'save_every': save_every}

        self.train_props = {'epochs': epochs, 'save_every': save_every}

        final_epoch = src_list.current_epoch + epochs
        super().__init__(src_list, NNModel([
            File(f.folder, f"epoch_{final_epoch:03d}.h5", f) for f in src_list.files
        ], self, src_list.model_props, final_epoch), props)

    def run(self, missing, *, parallel=None):
        if 'validate' in self.train_props:
            self.train_props['validate'].run(parallel=parallel)
        super().run(missing, parallel=parallel)

    def run_file(self, file):
        mkdir(file.folder)

        # with tf.profiler.experimental.Profile(str(LOG_FOLDER / 'tensorboard')):
        #     NNModelTrainer(file.folder, self.src_list, **self.train_props).train().clean()

        # with tf.profiler.experimental.Trace("StartTrain"):
        #     NNModelTrainer(file.folder, self.src_list, **self.train_props).train().clean()
        NNModelTrainer(file.folder, self.src_list, **self.train_props).train().clean()

        write_file(f"{file.path()}.json", json.dumps(self.export(), indent=2))


class NNModel(FileList):
    type = FileType.NN_MODEL
    current_epoch: int

    def __init__(self, files, task, model_props: ModelProperties, current_epoch=0):
        assert len(files) == 1
        self.model_props = model_props
        self.current_epoch = current_epoch
        super().__init__(files, task)

    def train(self, epochs: int, *, save_every=1) -> NNModel:
        """Train using fixed-size samples"""
        task = TrainNNModelTask(self, epochs, save_every)
        return task.dest_list

    def recognize(self, validate) -> ResultsJSONFileList:
        """Recognize using fixed-size samples"""
        task = RecognizeNNTask(self, validate)
        return task.dest_list

    def recognize_best_val_acc(self, testing) -> ResultsJSONFileList:
        self.run()
        b = find_best_models(self.files[0].folder).val_acc
        assert b is not None, "Best val acc model not found"
        task = RecognizeNNTask(self, testing, model_path=b.path, epoch=b.epoch)
        return task.dest_list

    def recognize_best_train_loss(self, testing) -> ResultsJSONFileList:
        self.run()
        b = find_best_models(self.files[0].folder).train_loss
        assert b is not None, "Best train loss model not found"
        task = RecognizeNNTask(self, testing, model_path=b.path, epoch=b.epoch)
        return task.dest_list

    def get_stats(self):
        self.run()
        return json.loads(read_file(f'{self.files[0].p}.stats.json'))

    def predict_best_val_acc(self, testing) -> PredictionResultList:
        self.run()
        b = find_best_models(self.files[0].folder).val_acc
        assert b is not None, "Best val acc model not found"
        task = PredictNNTask(self, testing, model_path=b.path, epoch=b.epoch)
        return task.dest_list
