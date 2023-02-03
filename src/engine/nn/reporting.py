import json
from os import path

from ..files.files import File, FileType
from ..files.lists import FileList
from ..files.tasks import TransformationTask
from ..helpers import read_file
from ..recognition import Result
from ..reporting import get_env_nn, save_results, ReportsFileList
from ..settings import RESULTS_FOLDER
from ..utils import mkdir, write_log, list_arg

"""NN Reporting

Author: Gilles Waeber <moi@gilleswaeber.ch>, VI 2019
"""


def extract_nn_results(recognition_log, dest_file, report_files: list, task):
    write_log(f"  Extract results from {recognition_log.path()}")
    print(recognition_log.path())
    records = [Result(**r) for r in json.loads(read_file(recognition_log.path()))]
    save_results(records, dest_file, report_files, task, get_env_nn())


class ExtractionJSONTask(TransformationTask):
    def __init__(self, src_list, report_name):
        report_name = list_arg(report_name)
        self.report_files = [RESULTS_FOLDER / r for r in report_name]
        super().__init__(src_list, ReportsFileList([
            File(f.folder, f"{f.p.stem}.report.json", f) for f in
            src_list.files
        ], self), {'report_name': report_name})

    def __str__(self):
        return 'Extract results'

    def run_file(self, file):
        mkdir(file.folder)
        extract_nn_results(file.task_src, path.splitext(file.path())[0], self.report_files, self)


class ResultsJSONFileList(FileList):
    type = FileType.RESULTS

    def __init__(self, files, task=None):
        super().__init__(files, task)

    def extract(self, report_name=None) -> ReportsFileList:
        task = ExtractionJSONTask(self, report_name)
        return task.dest_list

    def get(self, report_name=None):
        return self.extract(report_name).get()
