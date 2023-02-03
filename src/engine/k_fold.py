import datetime
import json
from dataclasses import dataclass
from os.path import commonpath
from pathlib import Path
from statistics import mean, stdev
from typing import List, Union

from .files.files import File
from .files.lists import FileList
from .files.parallel import ParallelFileList
from .files.tasks import VirtualTransformationTask, MergingTask, Task, find_task
from .helpers import write_file
from .hmm.recognition import ResultsFileList
from .nn.reporting import ResultsJSONFileList
from .reporting import flat_write, write_report, add_to_csv, ReportsFileList
from .settings import MODELS_FOLDER
from .settings import RESULTS_FOLDER
from .utils import print_log, write_log, list_arg


"""Transparent K-Fold cross-validation

Author: Gilles Waeber, VII 2019"""


@dataclass(frozen=True)
class TrainValTest:
    train: FileList
    val: FileList
    test: FileList


def save_nn_k_fold_results(test_results, stats, dest_file, report_files: list, task, env):
    test_acc = [r['accuracy'] for r in test_results]
    test_loss = [r['loss'] for r in test_results]
    val_acc = [r['val_acc'] for r in stats]
    val_loss = [r['val_loss'] for r in stats]
    epoch = [r['epoch'] for r in stats]
    print_log(f"  The accuracy is {mean(val_acc):.2%} ±{stdev(val_acc):.2%} for the validation "
              f"at epoch {mean(epoch):.1f} ±{stdev(epoch):.1f} - testing: {mean(test_acc):.2%} ±{stdev(test_acc):.2%}")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    result_common = {
        'id': str(Path(dest_file).relative_to(MODELS_FOLDER)),
        'timestamp': timestamp,
        'val_acc_m': mean(val_acc),
        'val_acc_d': stdev(val_acc),
        'val_loss_m': mean(val_loss),
        'val_loss_d': stdev(val_loss),
        'test_acc_m': mean(test_acc),
        'test_acc_d': stdev(test_acc),
        'test_loss_m': mean(test_loss),
        'test_loss_d': stdev(test_loss),
        'epoch_m': mean(epoch),
        'epoch_d': stdev(epoch),
    }

    result_structure = {
        **result_common,
        'env': env,
        'pipeline': task.export(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'epoch': epoch,
    }

    result_flat = {**result_common}
    flat_write(result_flat, 'env', env)
    flat_write(result_flat, 'pipeline', task.export())
    flat_write(result_flat, 'val_acc', val_acc)
    flat_write(result_flat, 'val_loss', val_loss)
    flat_write(result_flat, 'test_acc', test_acc)
    flat_write(result_flat, 'test_loss', test_loss)
    flat_write(result_flat, 'epoch', epoch)

    for report_file in report_files:
        write_report(report_file, result_flat, result_structure)

    # Write flat result file
    write_log(f"  Write flat results to {dest_file}.csv")
    add_to_csv(f"{dest_file}.csv", list(result_flat.keys()), [list(result_flat.values())])

    # Write structured result file
    write_log(f"  Write structured results to {dest_file}.json")
    write_file(f"{dest_file}.json", json.dumps(result_structure, indent=2))


def save_hmm_k_fold_results(test_results, val_stats, dest_file, report_files: list, task, env):
    test_acc = [r['accuracy'] for r in test_results]
    val_acc = [r['acc'] for r in val_stats]
    states = [r['states'] for r in val_stats]
    mixtures = [r['mixtures'] for r in val_stats]
    print_log(f"  The accuracy is {mean(val_acc):.2%} ±{stdev(val_acc):.2%} for the validation "
              f"with {mean(states):.1f} ±{stdev(states):.1f} states and "
              f"{mean(mixtures):.1f} ±{stdev(mixtures):.1f} mixtures - "
              f"testing: {mean(test_acc):.2%} ±{stdev(test_acc):.2%}")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    result_common = {
        'id': str(Path(dest_file).relative_to(MODELS_FOLDER)),
        'timestamp': timestamp,
        'val_acc_m': mean(val_acc),
        'val_acc_d': stdev(val_acc),
        'test_acc_m': mean(test_acc),
        'test_acc_d': stdev(test_acc),
        'states_m': mean(states),
        'states_d': stdev(states),
        'mixtures_m': mean(mixtures),
        'mixtures_d': stdev(mixtures),
    }

    result_structure = {
        **result_common,
        'env': env,
        'pipeline': task.export(),
        'val_acc': val_acc,
        'test_acc': test_acc,
        'mixtures': mixtures,
        'states': states,
    }

    result_flat = {**result_common}
    flat_write(result_flat, 'env', env)
    flat_write(result_flat, 'pipeline', task.export())
    flat_write(result_flat, 'val_acc', val_acc)
    flat_write(result_flat, 'test_acc', test_acc)
    flat_write(result_flat, 'mixtures', mixtures)
    flat_write(result_flat, 'states', states)

    for report_file in report_files:
        write_report(report_file, result_flat, result_structure)

    # Write flat result file
    write_log(f"  Write flat results to {dest_file}.csv")
    add_to_csv(f"{dest_file}.csv", list(result_flat.keys()), [list(result_flat.values())])

    # Write structured result file
    write_log(f"  Write structured results to {dest_file}.json")
    write_file(f"{dest_file}.json", json.dumps(result_structure, indent=2))


def get_kfold_set(item: Union[Task, FileList]):
    t: KFoldSeparationTask = find_task(item, KFoldSeparationTask)
    if t.n is None:
        raise ValueError('Fold unknown')
    return t.sets[t.n]


class KFoldParallelList(ParallelFileList):
    def create_nn_model(self, *args, **kwargs):
        return self.f('create_nn_model')(*args, pfl_inject={'validate': 'val'}, **kwargs)

    def train(self, *args, **kwargs):
        if 'epochs' in kwargs:
            pfl_suffix = f'_e{kwargs["epochs"]}'
        elif len(args) > 0:
            pfl_suffix = f'_e{args[0]}'
        else:
            pfl_suffix = '_eNA'
        return self.f('train')(*args, pfl_suffix=pfl_suffix, **kwargs)

    injectors = {
        'train': lambda d: d.train,
        'val': lambda d: d.val,
        'test': lambda d: d.test,
    }

    def evaluate_model(self, *args, **kwargs):
        return self.f('evaluate_model')(*args, pfl_inject={'validate': 'val'}, **kwargs)

    def recognize_best_val_acc(self, *args, **kwargs):
        return self.f('recognize_best_val_acc')(*args, pfl_suffix='_bva', pfl_inject={'testing': 'test'}, **kwargs)

    def recognize_best_train_loss(self, *args, **kwargs):
        return self.f('recognize_best_val_acc')(*args, pfl_suffix='_btl', pfl_inject={'testing': 'test'}, **kwargs)

    def extract(self, report_name=None):
        """This will merge the K-Fold results"""
        if isinstance(self.lists[0][0], ResultsJSONFileList):
            task = KFoldNNMergingTask(self, report_name)
            return task.dest_list
        elif isinstance(self.lists[0][0], ResultsFileList):
            task = KFoldHMMMergingTask(self, report_name)
            return task.dest_list

    def get(self, report_name=None):
        return self.extract(report_name).get()

    def k_fold(self, *args, **kwargs):
        raise ValueError('Nested K-Fold attempted')

    def features_digest(self, *args, **kwargs):
        raise ValueError('Not allowed')


class KFoldSeparationTask(VirtualTransformationTask):
    def __init__(self, src_list: FileList, k: int, val_bins=1, test_bins=1, n=None):

        assert k >= 3, f"Invalid k: {k}"
        assert val_bins + test_bins < k, f"Invalid number of bins: {val_bins} + {test_bins}"
        self.by_label = src_list.by_label()
        min_label = len(min(self.by_label.values(), key=len))
        self.per_class_per_bin = min_label // k
        self.per_class_in_bins = self.per_class_per_bin * k
        self.k = k
        self.n = n
        self.val_bins = val_bins
        self.test_bins = test_bins
        self.non_train_bins = val_bins + test_bins
        self.train_bins = k - self.non_train_bins
        self.remainder = []
        self.bins = [[] for _ in range(k)]

        for l, f in self.by_label.items():
            for b, bin_list in enumerate(self.bins):
                bin_list.extend(f[b:self.per_class_in_bins:k])
            self.remainder.extend(f[self.per_class_in_bins:])

        self.sets = []
        for i in range(k):
            va_bins = [self.bins[l % k] for l in range(i, i + val_bins)]
            te_bins = [self.bins[l % k] for l in range(i + val_bins, i + self.non_train_bins)]
            tr_bins = [self.bins[l % k] for l in range(i + self.non_train_bins, i + k)]
            src_task = KFoldSeparationTask(src_list=src_list, k=k, val_bins=val_bins, test_bins=test_bins,
                                           n=i) if n is None else self
            self.sets.append(TrainValTest(
                val=src_list.__class__([f for l in va_bins for f in l], task=src_task),
                test=src_list.__class__([f for l in te_bins for f in l], task=src_task),
                train=src_list.__class__([f for l in tr_bins for f in l] + self.remainder, task=src_task),
            ))

        if n is None:
            super().__init__(src_list, KFoldParallelList([
                (s.train, s) for s in self.sets
            ], self, self, f'k{k}_v{val_bins}t{test_bins}'), properties={
                'k': k,
                'val_bins': val_bins,
                'test_bins': test_bins
            })
        else:
            super().__init__(src_list, self.sets[n].train, dict(k=k, val_bins=val_bins, test_bins=test_bins, n=n))
        if n is None:
            print_log(str(self))
        else:
            write_log(str(self))

    def __str__(self):
        num_classes = len(self.by_label)
        sum_remainder = sum((len(l) - self.per_class_in_bins) for l in self.by_label.values())
        per_bin = num_classes * self.per_class_per_bin
        f_va, f_te, f_tr = per_bin * self.val_bins, per_bin * self.test_bins, per_bin * self.train_bins + sum_remainder
        return (f'K-Fold with K={self.k} for {"+".join(str(len(l)) for l in self.by_label.values())} samples'
                f', {self.k} bins of {"+".join(str(self.per_class_per_bin) for i in range(len(self.by_label)))}'
                f' and a remainder of {"+".join(str(len(l) - self.per_class_in_bins) for l in self.by_label.values())}'
                f', for each fold: {f_tr} training, {f_va} validation, and {f_te} testing samples'
                + (f', experiment {self.n} only' if self.n is not None else ''))


class KFoldNNMergingTask(MergingTask):
    def __init__(self, src_list: KFoldParallelList, report_name: Union[List[str], str, None]):
        assert isinstance(src_list.root, KFoldSeparationTask)
        assert isinstance(src_list.lists[0][0], ResultsJSONFileList)

        report_name = list_arg(report_name)
        self.report_files = [RESULTS_FOLDER / r for r in report_name]

        src_results = [l for l, d in src_list.lists]
        base_path = Path(commonpath(f.p for l in src_results for f in l.files))
        combined_name = '-'.join(l.task.get_model_task().src_list.features_digest()[:4] for l, _ in src_list.lists)
        super().__init__(src_list, ReportsFileList([
            File(base_path
                 / f'k-fold_{combined_name}'
                 / f'rec_{src_list.pfl_string}.report.json', task_src=src_results)
        ], self), {'report_name': report_name})

    def run_file(self, file: File):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        test_results = [r.get() for r in file.task_src]
        stats = [max(f.get_stats(), key=lambda s: s['val_acc']) for f, d in self.src_list.previous.lists]
        save_nn_k_fold_results(test_results, stats, file.p.with_suffix(''), report_files=self.report_files, task=self,
                               env=test_results[0]['env'])


class KFoldHMMMergingTask(MergingTask):
    def __init__(self, src_list: KFoldParallelList, report_name: Union[List[str], str, None]):
        assert isinstance(src_list.root, KFoldSeparationTask)
        assert isinstance(src_list.lists[0][0], ResultsFileList)

        report_name = list_arg(report_name)
        self.report_files = [RESULTS_FOLDER / r for r in report_name]

        src_results = [l for l, d in src_list.lists]
        base_path = Path(commonpath(f.p for l in src_results for f in l.files))
        combined_name = '-'.join(l.task.get_model_task().src_list.features_digest()[:4] for l, _ in src_list.lists)
        super().__init__(src_list, ResultsJSONFileList([
            File(base_path
                 / f'k-fold_{combined_name}'
                 / f'rec_{src_list.pfl_string}.report.json', task_src=src_results)
        ], self), {'report_name': report_name})

    def run_file(self, file: File):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        test_results = [r.get() for r in file.task_src]
        print(file.p)
        print([f.task.src_list.task.src_list.get_best() for f in file.task_src])
        val_stats = [f.task.src_list.task.src_list.get_best() for f in file.task_src]
        save_hmm_k_fold_results(test_results, val_stats, file.p.with_suffix(''), report_files=self.report_files,
                                task=self,
                                env=test_results[0]['env'])
