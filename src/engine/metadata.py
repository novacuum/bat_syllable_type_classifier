import csv
import json, datetime, math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

from engine.helpers import read_file
from .settings import BIRDVOICE_FOLDER


"""Samples metadata handling

Author: Gilles Waeber, VII 2019"""


class Metadata:
    """Audio sample metadata"""
    __slots__ = ('props',)
    label: str
    type: str
    duration: float
    size: int
    peaks: Optional[List[float]]

    def __init__(self, label, **props):
        self.props = {'label': label, **props}

    def __getattr__(self, item):
        if item != 'props' and item in self.props:
            return self.props[item]
        else:
            raise AttributeError

    def __contains__(self, item):
        return item in self.props

    def with_ind_label(self):
        """Change the label to one identifying only the individual"""
        n_props = {**self.props}
        n_props['label'] = f'ind{n_props["individual"]:02}'
        return Metadata(**n_props)

    def with_syllable_label(self, label):
        """Change the label to one identifying only the individual"""
        n_props = {**self.props, 'label': label}
        return Metadata(**n_props)

    def as_slice_with_label(self, label, start, end, source_file_stem):
        """Change the label to one identifying only the individual"""
        n_props = {**self.props, 'duration': end-start, 'label': label, 'start': start, 'end': end, 'source_file_stem': source_file_stem}
        return Metadata(**n_props)


class MetadataDB:
    """Source for metadata"""

    def __init__(self, db):
        self.db = db

    def for_file(self, file):
        return self.db[Path(file).stem]

    def props(self):
        return self.db


class FileMetadataDB(MetadataDB):
    def __init__(self, file):
        self.file = file
        if file[-3:] == 'csv':
            with open(f'{BIRDVOICE_FOLDER}/{file}') as csv_file:
                data = list(csv.reader(csv_file, delimiter=',', lineterminator="\n"))
        else:
            data = json.loads(read_file(f'{BIRDVOICE_FOLDER}/{file}'))

        super().__init__(dict((k, Metadata(**v)) for k, v in data.items()))

    def props(self):
        return self.file


def metadata_db(source) -> MetadataDB:
    from .files.files import File
    if isinstance(source, MetadataDB):
        return source
    elif isinstance(source, dict):
        return MetadataDB(source)
    elif isinstance(source, str):
        return FileMetadataDB(source)
    elif isinstance(source, list) and (not len(source) or isinstance(source[0], File)):
        return MetadataDB(dict((f.p.stem, f.metadata) for f in source if f.metadata is not None))
    else:
        raise ValueError(f"Invalid metadata DB source: {type(source)}")
