import os
from collections import defaultdict
from os import path
from typing import Optional, Sequence

from tqdm import tqdm

from engine.settings import BIRDVOICE_DATA_DIR
from .files import File
from .tasks import TransformationTask, Task
from ..utils import mkdir, copy, print_log, list_ellipsis


"""Lists of files

Author: Gilles Waeber, VI 2019"""


class FileList:
    """A file list represents a list of files that are required to do a certain task"""
    files: Sequence[File]
    task: Optional[Task]
    type: str
    has_been_processed: bool

    def __init__(self, files, task):
        self.files = files
        self.task = task
        self.has_been_processed = False
        self.type = self.__class__.type

    def __str__(self):
        lst = list(str(f) for f in self.files)
        lst.sort()
        return '\n'.join(lst)

    def __len__(self):
        return len(self.files)

    def separate(self):
        parts = [
            self.__class__([f], self.task) for f in self.files
        ]
        for e in parts:
            e.has_been_processed = self.has_been_processed
        return parts

    def by_label(self):
        by_label = defaultdict(list)
        for e in self.files:
            by_label[e.metadata.label].append(e)
        return by_label

    def copy(self, folder_suffix):
        """Copy to a named subfolder"""
        task = CopyTask(self, folder_suffix)
        return task.dest_list

    def get_missing(self):
        return [f for f in self.files if not path.exists(f.path())]

    def run(self, *, parallel=None):
        if self.has_been_processed:
            return self
        missing = self.get_missing()
        if len(missing) > 0:
            if self.task is None:
                raise FileNotFoundError(';'.join(f.path() for f in missing))
            else:
                self.task.run(missing, parallel=parallel)
        else:
            print_log(
                f"Already processed: {list_ellipsis([str(f.p.relative_to(BIRDVOICE_DATA_DIR)) for f in self.files])}")
        self.has_been_processed = True
        return self

    def clear(self):
        """Clear the leaf artifacts"""
        assert self.task is not None and isinstance(self.task, TransformationTask), "The files are not generated"
        to_remove = list(set(f.path() for f in self.files if path.exists(f.path())))
        for f in tqdm(to_remove, disable=len(to_remove) < 10):
            os.remove(f)
        self.has_been_processed = False
        return self


class CopyTask(TransformationTask):
    def __init__(self, src_list, folder_suffix):
        super().__init__(src_list, src_list.__class__([
            File(f"{f.folder}{folder_suffix}", f.name, f) for f in src_list.files
        ], self), {'folder_suffix': folder_suffix})

    def run_file(self, file):
        mkdir(file.folder)
        copy(file.task_src.path(), file.path())
