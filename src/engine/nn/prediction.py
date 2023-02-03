import re, csv
from collections import defaultdict
import numpy as np
from tensorflow.python.keras.models import load_model, Model
from engine.features.feature_extraction import FeaturesFileList
from engine.files.files import FileType, File
from engine.files.lists import FileList
from engine.files.tasks import TransformationTask
from engine.nn.properties import ModelProperties
from engine.processing.audio.splitseq import SplitIntoPartsTask
from engine.recognition import Result
from engine.utils import gpu_free
from utils.file import expand_data_path, to_unix_path, to_local_data_path
from utils.task import find_task


def predict_nn(model_path, ffl: FeaturesFileList, dest_file, model_props: ModelProperties):
    """Recognize the samples in batch"""
    model: Model = load_model(model_path)
    files = ffl.files
    x, y, _ = ffl.get_xyw(model_props)
    y_pred = model.predict(x)
    y_pred_text = [model_props.num_label[z] for z in np.argmax(y_pred, axis=1)]
    results = [Result(f.path(), y, p, None) for f, p, y in zip(files, y_pred_text, y_pred)]

    split_task: SplitIntoPartsTask = find_task(ffl.task, SplitIntoPartsTask)
    start_offset = (split_task.props['part_length'] * (1 - split_task.props['label_min_cover_length']))/2

    p = re.compile('_(\d{6})_')
    current_row = {'label': results[0].predicted, 'start': 0}
    confidence = max(results[0].truth)
    rows = []
    print(split_task.props)

    for result in results:
        match = p.findall(result.filename)
        time = float(match[0]) / 1000

        if current_row['label'] == result.predicted:
            confidence = (confidence + max(result.truth)) / 2
            continue
        
        current_row['end'] = time + start_offset
        current_row['duration'] = current_row['end'] - current_row['start']
        current_row['confidence'] = confidence
        rows.append(current_row)

        confidence = max(result.truth)
        current_row = {'label': result.predicted, 'start': current_row['end']}

    with open(dest_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, dialect='excel', fieldnames=['label', 'start', 'end', 'duration', 'confidence'])
        writer.writeheader()
        writer.writerows(rows)

    gpu_free()


class PredictNNTask(TransformationTask):
    predict: FeaturesFileList

    def __init__(self, src_list, predict: FileList, model_path=None, epoch=None):
        assert (model_path is None) == (epoch is None)

        if predict.type == FileType.AUDIO:
            print('pre process')
            print(type(predict).__name__)
            predict = predict.preproc_from(src_list)
            print(type(predict).__name__)
        assert predict.type == FileType.FEATURES
        assert len(src_list.files) == 1
        if model_path is None:
            model = src_list.files[0]
            epoch = src_list.current_epoch
        else:
            model = File(expand_data_path(model_path))
        self.predict = predict
        self.model_props = src_list.model_props

        by_source_file_stem = defaultdict(list)
        for file in predict.files:
            by_source_file_stem[file.metadata.source_file_stem] = file

        super().__init__(src_list, PredictionResultList([
            File(f"{model.folder}/predict", f"predict_{stem}_e{epoch}.csv", model)
            for stem, files in by_source_file_stem.items()
        ], self), {'model_path': to_unix_path(to_local_data_path(model_path))})

    def run(self, missing, *, parallel=None):
        self.predict.run(parallel=parallel)
        super().run(missing)

    def run_file(self, file):
        file.p.parent.mkdir(exist_ok=True)
        predict_nn(file.task_src.path(), self.predict, file.path(), self.model_props)


class PredictionResultList(FileList):
    type = FileType.RESULTS

    def __init__(self, files, task=None):
        super().__init__(files, task)

