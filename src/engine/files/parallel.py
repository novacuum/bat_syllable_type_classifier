from __future__ import annotations

import math
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Any, Optional

from .files import FileType
from .lists import FileList
from .tasks import TransformationTask, VirtualTransformationTask
from ..utils import print_log

"""Base class for the processing of tasks parallel to each other, e.g. for K-Fold

Author: Gilles Waeber, VII 2019"""


class ParallelFileList:
    type = FileType.PARALLEL
    lists: List[Tuple[FileList, Any]]
    root: TransformationTask
    task: TransformationTask
    previous: Optional[ParallelFileList]
    pfl_suffix: str

    def __init__(self, lists: List[Tuple[FileList, Any]], root: TransformationTask, task: TransformationTask,
                 pfl_string: str, previous=None):
        self.lists = lists
        self.root = root
        self.task = task
        self.pfl_string = pfl_string
        self.previous = previous

    def __getattr__(self, attr):
        """Allows to use parallel lists transparently"""
        return self.f(attr)

    def f(self, attr):
        """Execute a function in parallel"""
        if callable(getattr(self.lists[0][0], attr, None)):
            def virtual_method(*args, pfl_suffix='', pfl_inject=None, **kwargs):
                print_log(f'Virtual method call: {attr}('
                          f'{", ".join(str(a) for a in args)}, '
                          f'{", ".join(f"{k}={v}" for k, v in kwargs.items())})')
                task = ParallelTask(self, attr, kwargs, pfl_suffix, pfl_inject, args)
                return task.dest_list

            return virtual_method
        elif attr == 'files':
            return sum((l.files for l, _ in self.lists), [])
        else:
            raise AttributeError(f'Invalid attribute: {attr}')

    def new(self, lists: List[Tuple[FileList, Any]], task: TransformationTask, pfl_suffix: str = ''):
        return self.__class__(lists=lists, root=self.root, task=task, pfl_string=f'{self.pfl_string}{pfl_suffix}',
                              previous=self)


class ParAct:

    def __init__(self, action, args, num):
        self.action = action
        self.args = args
        self.num = num

    def __call__(self, fl_data__kwargs):
        (fl, data), kwargs = fl_data__kwargs
        with ThreadPool(int(math.ceil(cpu_count() / self.num))) as p:
            return getattr(fl, self.action)(*self.args, parallel=p, **kwargs), data


class ParallelTask(VirtualTransformationTask):
    """Task executed in parallel over multiple lists"""

    def __init__(self, src_list, action, kwargs, pfl_suffix, pfl_inject, args=()):

        if 'parallel' in kwargs and hasattr(kwargs['parallel'], 'map'):
            parallel = kwargs.pop('parallel')
        else:
            parallel = None

        if pfl_inject is not None:
            print_log(f'Injected: {", ".join(f"{k}=..." for k, _ in pfl_inject.items())}')
            injections = [dict((k, src_list.injectors[v](d)) for k, v in pfl_inject.items()) for _, d in src_list.lists]
            call_kwargs = [{**kwargs, **ikw} for ikw in injections]
        else:
            pfl_inject = {}
            call_kwargs = [kwargs for _ in src_list.lists]

        if parallel is not None:
            new_list = parallel.map(ParAct(action, args, len(src_list.lists)), zip(src_list.lists, call_kwargs))
        else:
            new_list = [(getattr(fl, action)(*args, **kwargs), data) for (fl, data), kwargs in
                        zip(src_list.lists, call_kwargs)]
        props = dict((k, v) for k, v in new_list[0][0].task.props.items() if k not in pfl_inject)

        super().__init__(src_list, src_list.new(new_list, self, pfl_suffix), dict(
            action=action,
            kwargs=props,
            pfl_suffix=pfl_suffix,
            pfl_inject=pfl_inject
        ))
