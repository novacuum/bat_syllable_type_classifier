import re
from typing import Dict

from engine.files.tasks import Task


def find_task(task: Task, type):
    while hasattr(task, 'src_list'):
        task = task.src_list.task
        if isinstance(task, type):
            return task

    return None


def create_shortened_identifier(key):
    return ''.join(m.group(1) for m in re.finditer(r"(?:^|_)(\w)", key))


def create_shortened_identifier_with_value(join='_', **kwargs):
    dest_folder_suffix = []
    for key, value in kwargs.items():
        if isinstance(value, Dict):
            dest_folder_suffix.append(create_shortened_identifier_with_value(join, **value))
        else:
            dest_folder_suffix.append(create_shortened_identifier(key) + str(value))
    if len(dest_folder_suffix) == 0:
        return ''
    else:
        return join.join(dest_folder_suffix)
