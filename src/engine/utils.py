from __future__ import annotations

import datetime
import subprocess
import time
from os import listdir, path, makedirs
from pathlib import Path
from shutil import copyfile
from subprocess import Popen
from typing import List, TYPE_CHECKING

# noinspection PyUnresolvedReferences,PyPackageRequirements,PyCompatibility
import __main__

from .helpers import TmpFile
from .settings import LOG_FOLDER, LOG_FILE, BIN_SOX

if TYPE_CHECKING:
    from .files.files import File
    import numpy as np

"""Utilities tied to the BirdVoice engine

Author: Gilles Waeber, 2018"""


def run(name, fx):
    """Run a function and time it"""
    print_log('{}...'.format(name))
    timer = time.perf_counter()
    fx()
    print_log('{} done! ({:.3f} s)'.format(name, time.perf_counter() - timer))


def list_files(folder):
    """List files in a folder"""
    items = [f for f in listdir(folder) if path.isfile(path.join(folder, f))]
    items.sort()
    return items


def mkdir(folder):
    """Create folder if it doesn't exist"""
    if not path.isdir(folder):
        write_log(f'  mkdir {folder}')
        makedirs(folder)


def copy(src, dest, progress=None):
    """Copy a file"""
    if progress is not None:
        progress.step(f"  cp {src} {dest}")
    else:
        write_log(f"  cp {src} {dest}")
    with TmpFile(dest) as dest_tmp:
        copyfile(src, dest_tmp)


def print_log(message, color=None):
    """Write message to log and print it in stdout"""
    write_log(message)
    if color is not None:
        from termcolor import colored
        print(colored(message, color))
    else:
        print(message)


# Create the log file
if not LOG_FOLDER.is_dir():
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)
log_file_handle = LOG_FILE.open('w', buffering=1)
current_script = Path(__main__.__file__).stem if '__file__' in __main__.__dict__ else 'interactive'
print(f"Log for {current_script} will be written in {LOG_FILE.stem}")


def write_log(message):
    """Write message to log"""
    return log_file_handle.write(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {message}\n")


def list_ellipsis(items, limit=3):
    """Create a human readable list with ellipsis"""
    if len(items) == 0:
        return '[]'
    elif len(items) <= limit:
        return ', '.join(items)
    else:
        write_log(f"The full list contains {len(items)} elements: {', '.join(items)}")
        return f"{', '.join(items[0:limit])}, ... ({len(items) - limit} others)"


def list_arg(arg) -> list:
    """Convert an argument to a list if it is not a list. [] for None, [x] for x."""
    if arg is None:
        return []
    elif not isinstance(arg, list):
        return [arg]
    else:
        return arg


def list_to_ranges(l):
    """Convert a list of numbers to a string denoting ranges, e.g. 1,3,2,5 -> 1-3,5"""
    if not len(l):
        return ""
    ranges = []
    current_start = None
    current_stop = None
    for el in sorted(l):
        if current_stop is None or el > current_stop + 1:
            if current_stop is not None:
                if current_start == current_stop:
                    ranges.append(str(current_start))
                else:
                    ranges.append(f'{current_start}-{current_stop}')
            current_start, current_stop = el, el
        else:
            current_stop = el
    if current_start == current_stop:
        ranges.append(str(current_start))
    else:
        ranges.append(f'{current_start}-{current_stop}')
    return ','.join(ranges)


def write_mlf(*, dest_file, files: List[File]):
    """Write a Master Label File"""
    with TmpFile(dest_file) as out:
        with open(out, 'wb') as f:
            f.write(b'#!MLF!#\n')
            for file in files:
                f.write(f'"*/{file.p.stem}.lab"\n'.encode())
                f.write(f"{file.metadata.label}\n".encode())
                f.write(b'.\n')


def irange(start, stop, step=1):
    """Generate an inclusive range"""
    return list(range(start, stop + step, step))


class Progress:
    """Show progress"""

    def __init__(self, total=1):
        self.total = total
        self.progress = -1

    def step(self, message):
        self.progress += 1
        self.print(message)

    def print(self, message):
        p = 100 * self.progress / self.total
        write_log(message)
        print("\r  [{0:>5.1f}%] {1}  \r  [{0:>5.1f}%] ".format(
            p, message), end='')

    def done(self):
        self.progress = self.total
        print('\r  [100.0%] ')


def call(args, *, progress=None, print_stdout=True, shell=False):
    """Call an external program and return the output"""
    args = [str(a) for a in args]
    if progress is not None:
        progress.step(' '.join(args))
    else:
        write_log("  {}".format(' '.join(args)))

    # if shell is true, Popen needs a string and not a argument list
    p = Popen(' '.join(args) if shell else args, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = []
    for raw_line in p.stdout:
        output.append(raw_line)
        if print_stdout:
            line = raw_line.decode().rstrip()
            if progress is not None:
                progress.print(line)
            else:
                write_log(f"  {line}")
    status = p.wait()
    if status != 0:
        write_log(f"Subprocess exited with non-zero status: {status}")
        raise RuntimeError(f"Subprocess exited with non-zero status: {status}\n"
                           f"Command: {' '.join(args)}\nOutput:\n{b''.join(output).decode()}")
    return b''.join(output)


def call_sox(args, *, progress=None):
    """Run SOX Sound Toolbox"""
    call([BIN_SOX] + args, progress=progress)


def call_piped_sox(args, *, progress=None):
    call(map(lambda value: value.replace('[bin:sox]', str(BIN_SOX)) if isinstance(value, str) else value, args), progress=progress, shell=True)


def gpu_lazy_allocation():
    """Don't pre-allocate all memory on the GPU"""
    import tensorflow as tf

    if not tf.executing_eagerly():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)


def gpu_free():
    """Release GPU resources"""
    from tensorflow.keras import backend
    backend.clear_session()
    # aggressive method: it's not possible to reuse the GPU afterwards
    # from numba import cuda
    # cuda.select_device(0)
    # cuda.close()


def plt_im_show(img, cmap='gray', plt_show=True, colorbar=False, axes=False, interpolation='bicubic'):
    """Display an image using matplotlib"""
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap=cmap, interpolation=interpolation)  # also works for color pictures
    if not axes:
        ax.set_xticks([]), ax.set_yticks([])  # to hide tick values on X and Y axis
    if colorbar:
        fig.colorbar(im)
    fig.show()
    if plt_show:
        plt.show()


def hsv2rgb(hsv: np.ndarray) -> np.ndarray:
    """HSV to RGB color space conversion.

    hsv: (..., 3), numpy ndarray, the image in HSV format. Final dimension denotes channels.

    returns: (..., 3), numpy ndarray, the image in RGB format. Same dimensions as input.

    Raises:
    - ValueError: If `hsv` is not at least 2-D with shape (..., 3).

    Notes: Conversion between RGB and HSV color spaces results in some loss of precision, due to integer arithmetic
    and rounding.

    Source: skimage.color.hsv2rgb (scikit-image) modified BSD licence, author: Ralph Gommers.
    """
    import numpy as np

    if hsv.shape[-1] != 3:
        raise ValueError(f"Input array must have a shape == (..., 3)), got {hsv.shape}")

    hi = np.floor(hsv[..., 0] * 6)
    f = hsv[..., 0] * 6 - hi
    p = hsv[..., 2] * (1 - hsv[..., 1])
    q = hsv[..., 2] * (1 - f * hsv[..., 1])
    t = hsv[..., 2] * (1 - (1 - f) * hsv[..., 1])
    v = hsv[..., 2]

    hi = np.stack([hi, hi, hi], axis=-1).astype(np.uint8) % 6
    out = np.choose(
        hi, np.stack([np.stack((v, t, p), axis=-1),
                      np.stack((q, v, p), axis=-1),
                      np.stack((p, v, t), axis=-1),
                      np.stack((p, q, v), axis=-1),
                      np.stack((t, p, v), axis=-1),
                      np.stack((v, p, q), axis=-1)]))

    return out
