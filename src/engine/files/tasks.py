from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Union

from tqdm import tqdm

from ..utils import print_log

if TYPE_CHECKING:
    from ..files.files import File
    from ..files.lists import FileList


class Task:
    def __init__(self, dest_list, properties):
        self.dest_list = dest_list
        self.props = properties

    def __str__(self):
        return type(self).__name__

    def run(self, missing, *, parallel=None):
        raise NotImplementedError()

    def export(self):
        """Export this task and the previous tasks"""
        this_task = [{'task': type(self).__name__, 'props': self.props}]
        if isinstance(self, TransformationTask):
            return self.src_list.task.export() + this_task
        return this_task

    def get_source_task(self) -> SourceTask:
        """Get the source task for this chain"""
        if not isinstance(self, SourceTask):
            if not isinstance(self, TransformationTask):
                raise ValueError("There exists no source task or the chain is broken")
            return self.src_list.task.get_source_task()
        else:
            return self

    def get_model_task(self) -> CreateModelTask:
        """Get the model creation task for this chain"""
        if not isinstance(self, CreateModelTask):
            if not isinstance(self, TransformationTask):
                raise ValueError("There exists no model creation task or the chain is broken")
            return self.src_list.task.get_model_task()
        else:
            return self


class SourceTask(Task, ABC):
    """A task that loads external elements"""

    def __init__(self, name, dest_list, properties):
        super().__init__(dest_list, properties)
        self.name = name

    def run(self, missing, *, parallel=None):
        if len(missing):
            print_log(f"This is a source task, missing files: {', '.join(str(f.p) for f in missing)}")
            raise ValueError("This is a source task")


class TransformationTask(Task, ABC):
    """A task with one or more input and one or more outputs"""

    def __init__(self, src_list, dest_list, properties):
        super().__init__(dest_list, properties)
        self.src_list = src_list

    def run(self, missing, *, parallel=None):
        self.src_list.run(parallel=parallel)
        if len(missing) == 1:
            print_log(f'  {str(self)}')
        if hasattr(parallel, 'imap_unordered') and len(missing) > 1:
            list(tqdm(parallel.imap_unordered(self._run_file, missing)))
        else:
            for file in tqdm(missing, desc=str(self), disable=len(missing) == 1):
                self._run_file(file)

    def _run_file(self, file: File):
        if file.p.exists():
            print_log(f"  {file.path()} already exists")
        else:
            self.run_file(file)

    def run_file(self, file: File):
        raise NotImplementedError()


class CreateModelTask(TransformationTask, ABC):
    """A task that creates a new model"""

    def __init__(self, src_list, dest_list, properties, name):
        super().__init__(src_list, dest_list, properties)
        self.name = name


class VirtualTransformationTask(TransformationTask):
    """A task that does not create new files, only change the structure of the file list"""

    def run_file(self, file: File):
        raise ValueError(f'File missing for {str(self)}: {file.path()}')


class MergingTask(TransformationTask, ABC):
    """A task that merge all its inputs into a single file"""
    pass


class PreprocessingTask(TransformationTask, ABC):
    """A reusable preprocessing task"""

    def __init__(self, src_list, dest_list, properties):
        super().__init__(src_list, dest_list, properties)

    def reuse(self, src_list):
        """Reuse this transformation for another source"""
        return self.__class__(src_list, **self.props)


def find_task(item: Union[Task, FileList], task_type: type):
    """Find parent class of specific type"""
    from ..files.lists import FileList
    if isinstance(item, task_type):
        return item
    elif isinstance(item, TransformationTask):
        return find_task(item.src_list.task, task_type)
    elif isinstance(item, Task):
        raise ValueError(f'{task_type} Task not found')
    elif isinstance(item, FileList):
        return find_task(item.task, task_type)
    else:
        raise ValueError(f'Cannot use {type(item)}')
