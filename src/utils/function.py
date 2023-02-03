import inspect
import sys


def get_functions(module_name, prefix='exp_'):
    module = sys.modules[module_name]
    for function_name, function in inspect.getmembers(module, inspect.isfunction):
        if function_name.startswith(prefix) and callable(function):
            yield function_name, function
