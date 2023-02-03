from os import path
from typing import Union, List

from .reporting import ExtractionTask, ReportsFileList
from ..files.files import File, FileType
from ..files.lists import FileList
from ..files.tasks import TransformationTask
from ..helpers import write_lines, TmpFile
from ..metadata import metadata_db
from ..settings import BIN_HVite
from ..utils import mkdir, write_mlf, call, write_log

# ## Model files
# ## Recognition artifacts
# List of htk feature files for the testing
MODEL_VALIDATE_FEATURES = 'validate_features.lst'
# Master label file for the testing
MODEL_VALIDATE_MLF = 'valid.mlf'
# Recognition log
MODEL_RECOGNITION_LOG = 'valid.log'

"""Simple recognition

Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018
Perform a simple recognition

Usage:
  r = ... # Recognition results (see training.py)
"""


def recognize(
    *, mixtures,
    model_folder,
    src_validate_features,
    src_mlf,
    src_trained_mmf,
    dest_recognition_log,
):
    from .training import MODEL_WORDS_LIST, MODEL_HTK_CONF, MODEL_WNET, MODEL_SPELLING
    htk_conf_file = f"{model_folder}/{MODEL_HTK_CONF}"
    words_list_file = f"{model_folder}/{MODEL_WORDS_LIST}"
    src_word_net = f"{model_folder}/{MODEL_WNET}"
    src_spelling = f"{model_folder}/{MODEL_SPELLING}"

    # Run the HVite recognizer
    with TmpFile(dest_recognition_log) as out:
        call([
            BIN_HVite,
            '-C', htk_conf_file,
            '-i', out,
            '-H', src_trained_mmf(mixtures),
            '-w', src_word_net,
            '-I', src_mlf,
            '-S', src_validate_features, src_spelling, words_list_file
        ])


def recognize_task(dest_folder, model_file, src_validate):
    model_folder, _ = path.split(model_file)
    validate_features_file = f"{dest_folder}/{MODEL_VALIDATE_FEATURES}"
    validate_mlf_file = f"{dest_folder}/{MODEL_VALIDATE_MLF}"
    recognition_log_file = f"{dest_folder}/{MODEL_RECOGNITION_LOG}"

    # Write features list
    write_log(f"Write features list in {validate_features_file}")
    write_lines(validate_features_file,
                (f.path() for f in src_validate.files))

    # Write master label file
    write_log(f"Write MLF in {validate_mlf_file}")
    write_mlf(dest_file=validate_mlf_file, files=src_validate.files)

    recognize(
        mixtures=None,
        model_folder=model_folder,
        src_validate_features=validate_features_file,
        src_mlf=validate_mlf_file,
        src_trained_mmf=lambda x: model_file,
        dest_recognition_log=recognition_log_file,
    )


class RecognizeTask(TransformationTask):
    def __init__(self, src_list, validate):
        if validate.type == FileType.AUDIO:
            validate = validate.preproc_from(src_list)
        assert validate.type == FileType.FEATURES
        super().__init__(src_list, ResultsFileList([
            File(f"{f.folder}/rec_{validate.features_digest()}/{f.p.stem.replace('training_', '')}",
                 f"valid.log", f, metadata=metadata_db(validate.task.src_list.files)) for f in
            src_list.files
        ], self), {'validate': validate.task.export()})
        self.validate = validate

    def run(self, missing, *, parallel=None):
        self.validate.run(parallel=parallel)
        super().run(missing, parallel=parallel)

    def run_file(self, file):
        mkdir(file.folder)
        recognize_task(file.folder, file.task_src.path(), self.validate)


class ResultsFileList(FileList):
    type = FileType.RESULTS

    def __init__(self, files, task=None):
        super().__init__(files, task)

    def extract(self, report_name: Union[str, List[str]] = None) -> ReportsFileList:
        task = ExtractionTask(self, report_name)
        return task.dest_list

    def get(self, report_name=None):
        return self.extract(report_name).get()
