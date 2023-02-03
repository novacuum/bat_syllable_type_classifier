import re
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np

from engine.helpers import TmpFile
from engine.features.feature_extraction import ExtractGeoFeaturesTask, ExtractHOGHWRecogTask, FeaturesFileList, \
    ExtractGeoHWRecogTask, ExtractPixelsTask, ExtractHOGFeaturesTask
from .files.files import File, FileType
from .files.lists import FileList
from .files.tasks import PreprocessingTask, Task, find_task
from .utils import mkdir, call_sox, print_log

# import pgmagick as magick
# from pgmagick import Image

# When the spectrogram is slightly smaller, we repeat the last segment
SOX_WIDTH_TOLERANCE = 3

"""Create spectrograms

Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018

Usage:
    s = ...  # A spectrograms collection (see audio.py)
    sb = ...  # A binary spectrograms collection
    f = ...  # A feature files collection, Geo or HoG (see feature_extraction.py)
    threshold = 55 # Binarization threshold, in %

    sb = s.binarize(threshold) # Binarize the spectrograms
    sb = s.smart_binarize() # Binarize the spectrograms, automatic threshold
    f = sb.geo_features() # Extract geometric features from binarized spectrograms
    f = s.hog_features() # Extract HoG features from grayscale spectrograms"""


# Create spectrograms
def create_spectrogram(
    src_file, dest_file, *,
    sampling_rate=32000, x_pixels_per_sec=100, width=None, height=256, window='Hann',
    axes=False, color=False, light_background=False,
    norm=None
):
    with TmpFile(dest_file) as out:
        args = [
            src_file,
            '-n',  # SoX effects
            'remix', '1',  # Keep 1st channel only
        ]
        # Resample effect
        if sampling_rate is not None:
            args += ['rate', sampling_rate]

        # Normalization effect
        if norm is not None:
            args += ['norm', str(norm)]

        # Spectrogram effect
        args += ['spectrogram',
                 # '-d', '0:10', # Audio duration
                 '-w', window,  # Window function
                 '-o', out  # Output file
                 ]
        if not color:
            if light_background:
                args += ['-l']  # Light background
            args += ['-m']  # Monochrome
        if width is not None:
            args += ['-x', str(width)]  # Total width
        else:
            args += ['-X', str(x_pixels_per_sec)]  # X-axis pixels/second
        if height is not None:
            # if height > 0 and (height & (height-1) == 0):
            #     height += 1  # for optimal performance sox needs a height which is power of 2 + 1
            args += ['-y', str(height)]  # Height per channel

        if not axes:
            args += ['-r']  # Raw spectrogram; no axes or legends

        call_sox(args)

        if width is not None:
            # Check image size
            data = plt.imread(out, format='png')
            cur_w = data.shape[1]
            if cur_w == width:
                pass  # All good
            elif width - SOX_WIDTH_TOLERANCE <= cur_w < width:
                # Repeat the last element
                data = np.repeat(data, [1] * (cur_w - 1) + [width - cur_w + 1], axis=1)
                plt.imsave(out, data, format='png')
            else:
                raise ValueError(f'Width should be {width} but is {cur_w} (tolerance is {SOX_WIDTH_TOLERANCE})')


# Binarize spectrogram
#   threshold within the range 0.0-1.0
def binarize_spectrogram(src_file, dest_file, threshold, progress=None):
    if progress is not None:
        progress.step(f"convert {src_file} -threshold {threshold * 100 :.1f}% {dest_file}")
    img = Image(src_file)
    img.threshold(magick.Color.scaleDoubleToQuantum(threshold))
    img.write(dest_file)


def smart_binarize_spectrogram(src_file, dest_file, progress=None):
    if progress is not None:
        progress.step(
            f"convert {src_file} -monochrome {dest_file}")
    img = Image(src_file)
    img.quantizeColorSpace(magick.ColorspaceType.GRAYColorspace)
    img.quantizeColors(2)
    img.quantize()
    img.write(dest_file)


class CreateSpectrogramTask(PreprocessingTask):
    def __init__(self, src_list, **kwargs):

        dest_folder_suffix = []
        for key, value in kwargs.items():
            dest_folder_suffix.append(
                ''.join(m.group(1) for m in re.finditer(r"(?:^|_)(\w)", key)) + str(value))
        if len(dest_folder_suffix) == 0:
            dest_folder_suffix = ''
        else:
            dest_folder_suffix = '/' + '_'.join(dest_folder_suffix)

        super().__init__(src_list, SpectrogramFileList([
            File(f"{f.folder}{dest_folder_suffix}", f"{f.p.stem}.png", f) for f in src_list.files
        ], self), kwargs)

    def __str__(self):
        return 'Create spectrograms'

    def run_file(self, file):
        mkdir(file.folder)
        create_spectrogram(file.task_src.path(), file.path(), **self.props)


class BinarizeSpectrogramTask(PreprocessingTask):
    def __init__(self, src_list, threshold):
        super().__init__(src_list, SpectrogramBinFileList([
            File(f"{f.folder}/bi{threshold}", f.name, f) for f in src_list.files
        ], self), {'threshold': threshold})

    def __str__(self):
        return f"Binarize spectrograms with {self.props['threshold']}% threshold"

    def run_file(self, file: File):
        mkdir(file.folder)
        binarize_spectrogram(file.task_src.path(), file.path(), self.props['threshold'] / 100)


class SmartBinarizeSpectrogramTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, SpectrogramBinFileList([
            File(f"{f.folder}/sbi", f.name, f) for f in src_list.files
        ], self), {})

    def __str__(self):
        return f"Binarize spectrograms with automatic threshold"

    def run_file(self, file: File):
        mkdir(file.folder)
        smart_binarize_spectrogram(file.task_src.path(), file.path())


class SpectrogramBinFileList(FileList):
    type = FileType.SPECTROGRAM_BIN

    def __init__(self, files, task=None):
        super().__init__(files, task)

    def geo_features(self) -> FeaturesFileList:
        task = ExtractGeoFeaturesTask(self)
        return task.dest_list

    def geo_hwr_features(self) -> FeaturesFileList:
        task = ExtractGeoHWRecogTask(self)
        return task.dest_list


class SpectrogramFileList(FileList):
    type = FileType.SPECTROGRAM

    def __init__(self, files, task=None):
        super().__init__(files, task)

    def binarize(self, threshold) -> SpectrogramBinFileList:
        task = BinarizeSpectrogramTask(self, threshold)
        return task.dest_list

    def smart_binarize(self) -> SpectrogramBinFileList:
        task = SmartBinarizeSpectrogramTask(self)
        return task.dest_list

    def hog_hwr_features(self) -> FeaturesFileList:
        task = ExtractHOGHWRecogTask(self)
        return task.dest_list

    def hog_features(self) -> FeaturesFileList:
        task = ExtractHOGFeaturesTask(self)
        return task.dest_list

    def img_features(self) -> FeaturesFileList:
        """Raw pixel features"""
        task = ExtractPixelsTask(self)
        return task.dest_list


def get_x_per_sec(item: Union[Task, FileList]) -> Optional[float]:
    """Find the number of samples per second used during spectrogram extraction"""
    try:
        t: CreateSpectrogramTask = find_task(item, CreateSpectrogramTask)
        if 'x_pixels_per_sec' in t.props:
            return t.props['x_pixels_per_sec']
        else:
            return None
    except Exception as e:
        print_log(f'Failed to find x/sec: {e}', 'yellow')
        return None
