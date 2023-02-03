import json
from typing import List

from engine.features.feature_extraction import ExtractHOGHWRecogTask
from engine.files.tasks import Task
from engine.k_fold import get_kfold_set
from engine.processing.audio.splitseq import SplitIntoPartsTask
from engine.settings import BSC_ROOT_DATA_FOLDER
from engine.spectrograms import CreateSpectrogramTask, SpectrogramFileList
from model.pipeline import JsonPipeline
from utils.task import find_task


def fetch_filter_tasks(task: Task):
    result = []

    while hasattr(task, 'src_list'):
        task = task.src_list.task
        if 'Filter' in type(task).__name__:
            result.append(task)

    return result


class ReportResult:
    labels: List
    bin_size: int
    kfold_info: str
    spectrogram_images: SpectrogramFileList

    def __init__(self, json_result, result_name):
        from utils.report import get_best_model_pipeline_from_result
        self.result_name = result_name
        self.pipeline = JsonPipeline(json_result['pipeline'])
        self.model = get_best_model_pipeline_from_result(json_result)
        self.dataset_name = self.get_dataset_name()

        kfold = get_kfold_set(self.model)
        kfold_sepereation_taks = kfold.train.task
        spectrogram_task: CreateSpectrogramTask = find_task(self.model.task, CreateSpectrogramTask)

        self.bin_size = kfold_sepereation_taks.per_class_per_bin
        self.labels = kfold_sepereation_taks.by_label.keys()
        self.label_info = str(list(self.labels))
        self.spectrogram_images = spectrogram_task.dest_list
        self.spectrogram_info = self.create_config_info_by_task(spectrogram_task, {'sampling_rate': 32000, 'x_pixels_per_sec': 100, 'height': 256, 'window': 'Hann'})
        self.hog_info = self.create_config_info(ExtractHOGHWRecogTask, {'window_size': '16x265px', 'window_stride': '2px', 'block_size': '16x128px', 'block_stride': (16, 64), 'cell_size': (4,64), 'num_of_bins': 12})
        self.split_info = self.create_config_info(SplitIntoPartsTask, {})

        k_fold_n = kfold_sepereation_taks.n
        kfold_sepereation_taks.n = None
        self.kfold_info = str(kfold_sepereation_taks)
        kfold_sepereation_taks.n = k_fold_n

        filter_tasks = fetch_filter_tasks(self.model.task)
        if len(filter_tasks) > 0:
            self.filter_info = ['### Dataset filtering']
            self.filter_info.extend([
                f'- {type(task).__name__}: {self.create_config_info_by_task(task, {})}' for task in filter_tasks
            ])
            self.filter_info = '\n'.join(self.filter_info)
        else:
            self.filter_info = ''

    def get_model_path(self):
        return BSC_ROOT_DATA_FOLDER / self.dataset_name / 'models'

    def get_dataset_name(self):
        if self.pipeline.has_task('AudioLoadTask'):
            return self.pipeline.get_dataset_name()

        print(json.dumps(self.pipeline.pipeline, indent=2))
        raise Exception('no AudioLoadTask found in pipeline')

    def create_config_info(self, type, defaults):
        task: Task = find_task(self.model.task, type)
        if task:
            return self.create_config_info_by_task(task, defaults)

        return ''

    def create_config_info_by_task(self, task: Task, defaults):
        config_info = []
        for key, value in task.props.items():
            if key in defaults:
                del defaults[key]

            config_info.append(
                key.replace('_', ' ') + ' = ' + str(value))

        for key, value in defaults.items():
            config_info.append(
                key.replace('_', ' ') + ' = ' + str(value))

        return ', '.join(config_info)
