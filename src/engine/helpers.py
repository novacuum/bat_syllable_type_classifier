import os
from pathlib import Path


"""Utilities that are not tied to the BirdVoice engine

Author: Gilles Waeber, VI 2019"""


def read_lines(file, *, comment_char=None):
    """Read all lines from a file into an array, remove blank lines and lines starting with comment_char if set"""
    data = list(filter(lambda l: len(l) > 0, (line.rstrip() for line in open(file, encoding='utf-8'))))
    if comment_char is not None:
        data = [l for l in data if l[0] != comment_char]
    return data


def read_gz_file(file, *, mode='r'):
    import gzip
    with gzip.open(file, mode) as f:
        return f.read()


def read_file(file, *, mode='r', auto_unwrap=True):
    """Read a file"""
    if not isinstance(file, Path):
        file = Path(file)
    if auto_unwrap and file.suffix == '.gz':
        return read_gz_file(file, mode=mode)
    args = {} if 'b' in mode else dict(encoding='utf-8')
    with file.open(mode, **args) as f:
        return f.read()


def write_lines(file, lines):
    """Write lines to a file"""
    write_file(file, '{}\n'.format('\n'.join(lines)))


def write_gz_file(file, contents, *, mode='wt'):
    import gzip
    with TmpFile(file) as tmp, gzip.open(tmp, mode) as f:
        f.write(contents)


def write_file(file, contents, *, mode='wt', auto_wrap=True):
    """Write contents to a file"""
    if not isinstance(file, Path):
        file = Path(file)
    if auto_wrap and file.suffix == '.gz':
        return write_gz_file(file, contents, mode=mode)
    args = {} if 'b' in mode else dict(encoding='utf-8')
    with TmpFile(file) as tmp, open(tmp, mode, **args) as f:
        f.write(contents)


class TmpFile:
    """Gives a temp filename. Will rename the temp file to the correct name when no exception occurs."""

    def __init__(self, filename) -> None:
        self.filename = filename
        self.temp_filename = f'{filename}.{os.getpid()}.tmp'

    def __enter__(self) -> str:
        return self.temp_filename

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            os.replace(self.temp_filename, self.filename)
        elif os.path.isfile(self.temp_filename):
            os.remove(self.temp_filename)
        return False  # Do not suppress exceptions


class BoundingBox:
    """A simple bounding box"""

    def __init__(self, y, x, ylim=(-1, 1), xlim=(-1, 1)):
        self.down, self.up = y
        self.left, self.right = x
        self.y_min, self.y_max = ylim
        self.x_min, self.x_max = xlim

    def copy(self):
        return BoundingBox(
            (self.down, self.up),
            (self.left, self.right),
            (self.y_min, self.y_max),
            (self.x_min, self.x_max)
        )

    def ylim(self, y_min, y_max):
        self.y_min = y_min
        self.y_max = y_max

    def xlim(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def y(self, y_value):
        return self.down + self.height() * \
               (y_value - self.y_min) / self.y_height()

    def y_rev(self, y_position):
        return self.y_min + self.y_height() * \
               (y_position - self.down) / self.height()

    def x(self, x_value):
        return self.left + self.width() * \
               (x_value - self.x_min) / self.x_width()

    def x_rev(self, x_position):
        return self.x_min + self.x_width() * \
               (x_position - self.left) / self.width()

    def rev(self, xy):
        return self.x_rev(xy[0]), self.y_rev(xy[1])

    def pos(self, x_value, y_value):
        return self.x(x_value), self.y(y_value)

    def width(self):
        return self.right - self.left

    def x_width(self):
        return self.x_max - self.x_min

    def height(self):
        return self.up - self.down

    def y_height(self):
        return self.y_max - self.y_min
