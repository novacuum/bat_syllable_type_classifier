import csv
import datetime
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from filelock import FileLock

from utils.file import to_unix_path, to_local_data_path
from .files.files import FileType
from .files.lists import FileList
from .helpers import read_file, write_file, TmpFile
from .settings import BIN_SOX, BIN_HVite, BIN_HWRECOG, MODELS_FOLDER
from .utils import write_log, call, print_log


def get_sox_version():
    ver_string = call([BIN_SOX, '--version'], print_stdout=False).decode()
    match = re.search(r"SoX\s+v([\d.]+)\s*$", ver_string)
    assert match, f"Failed to extract SoX version from {ver_string}"
    return match.group(1)


def get_htk_version():
    ver_string = call([BIN_HVite, '-V'], print_stdout=False).decode()
    match = re.search(r"HVite\s+(\S+)\s", ver_string)
    assert match, f"Failed to extract HVite version from {ver_string}"
    return match.group(1)


def get_java_version():
    ver_string = call(['java', '-version'], print_stdout=False).decode()
    return [l for l in re.split(r"[\r\n]+", ver_string) if len(l)]


def get_env_htk():
    """Get environment information for HTK"""
    import platform
    return {
        'os': platform.system(),
        'platform': platform.platform(),
        'sox': get_sox_version(),
        'htk': get_htk_version(),
        'hwrecog': BIN_HWRECOG.name,
        'java': get_java_version()
    }


def get_env_nn():
    """Get environment information for NN"""
    import tensorflow
    import platform
    from tensorflow.python.client import device_lib
    return {
        'os': platform.system(),
        'platform': platform.platform(),
        'sox': get_sox_version(),
        'java': get_java_version(),
        'keras': tensorflow.keras.__version__,
        'tf': tensorflow.__version__,
        'gpu': ', '.join(x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU')
    }


def flatten_dict(data, keep_null=False):
    dest = {}
    for k, v in data.items():
        flat_write(dest, k, v, keep_null)
    return dest


def flat_write(dest, key, value, keep_null=False):
    """Write data to a flat dict"""
    if isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
        dest[key] = value
    elif value is None:
        if keep_null:
            dest[key] = "null"
    elif isinstance(value, Path):
        dest[key] = str(value)
    elif isinstance(value, dict):
        for prop, value in value.items():
            flat_write(dest, f"{key}.{prop}", value)
    elif isinstance(value, list) and len(value) and isinstance(value[0], dict) and 'task' in value[0]:
        for i, task in enumerate(value):
            dest[f"{key}.{task['task']}"] = i + 1
            flat_write(dest, f"{key}.{task['task']}", task['props'])
    elif isinstance(value, list) or isinstance(value, tuple):
        for i, value in enumerate(value):
            flat_write(dest, f"{key}.{i + 1}", value)
    else:
        print_log(f"Unknown type for {key}: {type(value)}")
        raise RuntimeError(f"Unknown type for {key}: {type(value)}")


def save_results(records, dest_file, report_files: list, task, env):
    correct = len([r for r in records if r.is_correct])
    total = len(records)
    incorrect = total - correct
    accuracy = correct / total
    losses = [r.loss for r in records if r.loss is not None]
    loss = np.average(losses) if len(losses) else None
    print_log(f"  The accuracy is {accuracy:.2%} ({correct}/{total}) - loss: {loss}")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    confusion_matrix = defaultdict(lambda: defaultdict(lambda: 0))
    for report_file in records:
        confusion_matrix[report_file.truth][report_file.predicted] += 1

    all_labels = set(confusion_matrix.keys())
    for v in confusion_matrix.values():
        all_labels = all_labels.union(v.keys())
    all_labels = list(all_labels)
    all_labels.sort()

    result_common = {
        'id': str(Path(dest_file).relative_to(MODELS_FOLDER)),
        'timestamp': timestamp,
        'accuracy': accuracy,
        'loss': loss,
        'correct': correct,
        'incorrect': incorrect,
        'total': total
    }

    result_structure = {
        **result_common,
        'confusion': dict(
            (k, dict({k: confusion_matrix[k][k]}, **dict(
                (k2, confusion_matrix[k][k2]) for k2 in all_labels if k2 != k)))
            for k in all_labels),
        'failed': [to_unix_path(to_local_data_path(r.filename)) for r in records if not r.is_correct],
        'env': env,
        'pipeline': task.export()
    }

    result_flat = {**result_common}
    for k in all_labels:
        # True positives
        result_flat[f"c.{k}.tp"] = confusion_matrix[k][k]
        # False negatives
        result_flat[f"c.{k}.fn"] = sum(confusion_matrix[k][i] for i in all_labels if i != k)
        # False positives
        result_flat[f"c.{k}.fp"] = sum(confusion_matrix[i][k] for i in all_labels if i != k)
    flat_write(result_flat, 'env', env)
    flat_write(result_flat, 'pipeline', task.export())

    for report_file in report_files:
        write_report(report_file, result_flat, result_structure)

    # Write flat result file
    write_log(f"  Write flat results to {dest_file}.csv")
    with TmpFile(f"{dest_file}.csv") as out, open(out, 'w', newline='') as f:
        # f.write('\uFEFF')  # UTF-8 BOM, Excel does not support this when sep is used
        f.write('"sep=,"\r\n')  # Field separator
        writer = csv.writer(f, dialect='excel')
        writer.writerow(result_flat.keys())
        writer.writerow(result_flat.values())

    # Write structured result file
    write_log(f"  Write structured results to {dest_file}.json")
    write_file(f"{dest_file}.json", json.dumps(result_structure, indent=2))


def add_to_csv(dest_file: str, headers: Sequence[str], values: Sequence[Sequence[str]]):
    with FileLock(f"{dest_file}.lock"):
        if os.path.isfile(f"{dest_file}"):
            with open(f"{dest_file}", 'r', newline='') as f:
                file_header = f.readline()
                assert re.match(r'^\W*"sep=,"\W*$', file_header), f"Invalid existing report file {file_header}"
                reader = csv.reader(f, dialect='excel')
                lines = list(reader)
                old_headers = lines[0]
                old_data = lines[1:]
                added_headers = []
                for h in headers:
                    if h not in old_headers:
                        added_headers.append(h)
                new_headers = old_headers + added_headers

                # Pad old data
                if len(added_headers) > 0:
                    old_data = [l + [None] * len(added_headers) for l in old_data]

                # Reorder new data
                title_old_num = dict((h, i) for i, h in enumerate(headers))
                old_nums = [(title_old_num[t] if t in title_old_num else None) for t in new_headers]
                new_data = [[(row[i] if i is not None else None) for i in old_nums] for row in values]

                headers = new_headers
                values = old_data + new_data
        with TmpFile(f"{dest_file}") as out, open(out, 'w', newline='') as f:
            # f.write('\uFEFF')  # UTF-8 BOM, Excel does not support this when sep is used
            f.write('"sep=,"\r\n')  # Field separator
            writer = csv.writer(f, dialect='excel')
            writer.writerow(headers)
            writer.writerows(values)


def write_report(report_file, result_flat, result_structured):
    # Write flat report file
    Path(report_file).parent.mkdir(exist_ok=True)
    headers = result_flat.keys()
    values = [[result_flat[k] for k in headers]]
    write_log(f"  Write flat report to {report_file}.csv")

    add_to_csv(f'{report_file}.csv', headers, values)

    # Write structured report file
    report = {'results': {}}
    with FileLock(f"{report_file}.json.lock"):
        if os.path.isfile(f"{report_file}.json"):
            report = json.loads(read_file(f"{report_file}.json"))
            assert report['results'], "Invalid report file"
        report['results'][result_structured['id']] = result_structured
        write_log(f"  Write structured report to {report_file}.json")
        write_file(f"{report_file}.json", json.dumps(report, indent=2))


class ReportsFileList(FileList):
    type = FileType.RESULTS

    def __init__(self, files, task=None):
        super().__init__(files, task)

    def get(self) -> dict:
        """Get report, triggers a run"""
        self.run()
        assert len(self.files) == 1, "Not handled: multiple report files"
        report = json.loads(read_file(self.files[0].path()))
        return report
