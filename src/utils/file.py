import os
from pathlib import Path
from typing import Union

from engine.files.files import File
from engine.settings import BSC_ROOT_DATA_FOLDER


def to_unix_path(folder_or_path: Union[Path, str, None]):
    if folder_or_path is None:
        return None

    return str(folder_or_path).replace(os.sep, '/')


def to_local_data_path(folder_or_path: Union[Path, str, None]):
    if folder_or_path is None:
        return None

    if not isinstance(folder_or_path, Path):
        folder_or_path = Path(folder_or_path)

    if is_path_relative_to(str(folder_or_path), str(BSC_ROOT_DATA_FOLDER)):
        return folder_or_path.relative_to(BSC_ROOT_DATA_FOLDER)

    return folder_or_path


def expand_data_path(folder_or_path: Union[Path, str]):
    if not isinstance(folder_or_path, Path):
        folder_or_path = Path(folder_or_path)

    if is_path_relative_to(str(folder_or_path), str(BSC_ROOT_DATA_FOLDER)):
        return folder_or_path

    return BSC_ROOT_DATA_FOLDER / folder_or_path


def is_path_relative_to(path, root):
    return path[0:len(root)] == root


def file_has_labels(file: File):
    return file.metadata is not None and 'labels' in file.metadata
