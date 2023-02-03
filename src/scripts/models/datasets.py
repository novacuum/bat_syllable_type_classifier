import csv
from dataclasses import dataclass
from typing import Sequence

from engine.audio import load_audio
from engine.metadata import metadata_db
from engine.settings import FILES_FOLDER

"""Woodcock datasets for the experiments

Author: Gilles Waeber, VII 2019"""


@dataclass(frozen=True)
class Dataset:
    name: str
    classes: Sequence[str]
    path: str




