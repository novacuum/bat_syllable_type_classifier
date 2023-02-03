from os import path

from .settings import FILES_FOLDER
from .utils import write_log
from engine.helpers import read_lines

"""Configuration

Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018
A configuration is a file describing a list of audio samples, located in the conf folder

Usage:
  from .configuration import conf
  c = ...  # A configuration filter (used in audio.py)

  file = 'training-ids.txt'  # A file containing a line-separated list of files (extensions are ignored), located in
  the conf folder
  c = conf(file) # Create a configuration filter
"""

def to_conf_file_path(dataset_name, source):
    return FILES_FOLDER / dataset_name / 'config' / source


def conf(dataset_name, source):
    if isinstance(source, Configuration):
        return source
    else:
        return Configuration(dataset_name, source)


class ConfigurationLine:
    def __init__(self, line):
        parts = line.split('\t')
        self.file = parts[0]
        if len(parts) > 1:
            self.repeat = int(parts[1])
            assert self.repeat >= 0, f"Invalid repeat count in '{line}'"
        else:
            self.repeat = 1


class Configuration:
    def __init__(self, dataset_name, source):
        file = to_conf_file_path(dataset_name, source)
        lines = read_lines(file, comment_char='#')
        self.list = [ConfigurationLine(line) for line in lines]
        self.file = source
        self.name = path.splitext(path.basename(self.file))[0]

    def props(self):
        return self.file

    def load(self):
        missing = []
        tracks_list = []
        for c in self.list:
            if not path.isfile(f"{FILES_FOLDER}/{c.file}"):
                missing.append(c.file)
                continue

            for i in range(c.repeat):
                tracks_list.append(f"{FILES_FOLDER}/{c.file}")

        if len(missing) > 0:
            write_log(f"Following tracks listed in {self.file} could not be found: {', '.join(missing)}")
            raise RuntimeError(f"Following tracks listed in {self.file} could not be found: {', '.join(missing)}")

        return tracks_list
