import re
from os import path
from typing import List, Union

from ..files.files import File
from ..files.tasks import TransformationTask
from ..metadata import MetadataDB
from ..recognition import Result
from ..reporting import get_env_htk, save_results, ReportsFileList
from ..settings import RESULTS_FOLDER
from ..utils import mkdir, write_log, list_arg
from engine.helpers import read_lines

"""Reporting

Author: Gilles Waeber <moi@gilleswaeber.ch>, XII 2018

Extract the results and collect them

Usage:
    r = ... # Recognition results (see recognition.py)
"""


class HTKResult(Result):
    def __init__(self, lines, mdb: MetadataDB):
        assert len(lines) > 0, "Empty record"
        match = re.search(r"[\\/]([^.\\/]+)\.rec\"", lines[0])
        assert match, f"Failed to extract filename from '{lines[0]}'"
        filename = match.group(1)
        truth = mdb.for_file(filename).label
        self.found_words = [HTKResultWord(l) for l in lines[1:]]
        predicted = ' '.join(m.label for m in self.found_words) or '(none)'

        super().__init__(filename, truth, predicted)


class HTKResultWord:
    """One word found during recognition"""

    def __init__(self, line):
        match = re.search(r"^(?P<start>\d+)\s+"
                          r"(?P<end>\d+)\s+"
                          r"(?P<label>[^ ]+)\s+"
                          r"(?P<probability>[\d.-]+)\s*$", line)
        assert match, f"Failed to extract match from '{line}'"
        self.start = match.group('start')
        self.end = match.group('end')
        self.label = match.group('label')
        self.probability = match.group('probability')


def read_recognition_log(recognition_log: File):
    """Parse the HVite recognition log"""

    lines = read_lines(recognition_log.path())
    # First pass: separate in records
    assert len(lines) and lines[0] == '#!MLF!#', f'Failed to parse {recognition_log.path()}'
    records = []
    record = []
    for line in lines[1:]:
        if line == '.':
            records.append(record)
            record = []
        else:
            record.append(line)
    # Second pass: parse the records
    records = [HTKResult(r, recognition_log.metadata) for r in records]
    return records


def extract_results(recognition_log, dest_file, report_files: list, task):
    write_log(f"  Extract results from {recognition_log.path()}")
    records = read_recognition_log(recognition_log)
    save_results(records, dest_file, report_files, task, get_env_htk())


class ExtractionTask(TransformationTask):
    def __init__(self, src_list, report_name: Union[List[str], str, None]):
        report_name = list_arg(report_name)
        self.report_files = [f"{RESULTS_FOLDER}/{r}" for r in report_name]
        super().__init__(src_list, ReportsFileList([
            File(f.folder, f"{f.p.stem}.json", f) for f in
            src_list.files
        ], self), {'report_name': report_name})

    def __str__(self):
        return 'Extract results'

    def run_file(self, file):
        mkdir(file.folder)
        extract_results(file.task_src, path.splitext(file.path())[0], self.report_files, self)
