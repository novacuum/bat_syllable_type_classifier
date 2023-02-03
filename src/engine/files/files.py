from __future__ import annotations

import os
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Union, Any

from ..metadata import MetadataDB, Metadata, metadata_db

"""File objects

Author: Gilles Waeber, 2018"""


class FileType(Enum):
    MIXED = -1
    NONE = 0
    AUDIO = 2
    CONF = 4
    PROFILE = 5
    SPECTROGRAM = 6
    SPECTROGRAM_BIN = 7
    FEATURES = 9
    MODEL = 10
    BLANK_MODEL = 11
    RESULTS = 12
    RNN_MODEL = 13
    NN_MODEL = 14
    PARALLEL = 15
    EVALUATION = 16


@total_ordering
class File:
    __slots__ = ('metadata', 'folder', 'name', 'task_src', 'p')
    metadata: Union[Metadata, MetadataDB]
    folder: str
    name: str
    p: Path
    task_src: Union[File, list, None]

    """A file that is the result of an operation

    The source of the file is either one or multiple files. It can also have associated metadata. The metadata from the
    source file is passed onto the resulting file unless overwritten.
    """

    counter = 0

    def __init__(self,
                 folder_or_path: Union[Path, str],
                 name: str = None,
                 task_src: Union[File, list, Any, None] = None, *,
                 metadata=None
                 ):
        File.counter += 1
        if isinstance(folder_or_path, Path):
            self.p = folder_or_path
            self.folder = str(folder_or_path.parent)
            self.name = folder_or_path.name
            assert name is None
        else:
            if name is None:
                folder_or_path, name = os.path.split(folder_or_path)
            self.folder = folder_or_path
            self.name = name
            self.p = Path(self.folder) / self.name

        self.task_src = task_src
        if metadata is not None:
            self.metadata = metadata
        elif isinstance(task_src, File):
            self.metadata = task_src.metadata
        elif isinstance(task_src, list) and len(task_src) and isinstance(task_src[0], File):
            self.metadata = metadata_db(task_src)
        else:
            self.metadata = None

    def __str__(self):
        return self.path()

    def __lt__(self, other):
        return self.p < other.p

    def __eq__(self, other):
        if not isinstance(other, File):
            return False
        return self.p == other.p

    def __hash__(self):
        return hash(self.p)

    def path(self):
        return f"{self.folder}/{self.name}"
