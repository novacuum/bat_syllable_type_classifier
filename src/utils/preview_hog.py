from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from engine.features.feature_sequence import FeatureSequence
import matplotlib.pyplot as plt

from engine.utils import hsv2rgb


def run(files):
    for i, file in enumerate(files):
        f = Path(file)

        if i > 0:
            plt.figure(i)
        fig, axes = plt.subplots(nrows=7, figsize=(18, 6), sharex='all')
        fig.canvas.set_window_title(f"{f.name} ({f.parent.absolute()})")
        plt.subplots_adjust(left=0.04,
                            bottom=0.085,
                            right=0.995,
                            top=0.99,
                            wspace=0,
                            hspace=0.04)

        fs = FeatureSequence.from_htk(f)
        if fs.x_per_sec is None:
            fs.x_per_sec = 1

        hog_raw_fig(fs, axes[0])
        hog_blocks_val(fs, axes[1])
        hog_blocks_dir(fs, axes[2])
        hog_cells_val(fs, axes[3])
        hog_cells_dir(fs, axes[4])
        hog_cells_combined(fs, axes[5])
        hog_cells_combined_unsigned(fs, axes[6])

    plt.show()


def hog_raw_fig(fs: FeatureSequence, ax: Axes):
    fs_data = fs.np_sequence.T
    im = ax.imshow(fs_data, extent=(0, fs.duration, 0, fs_data.shape[0]), aspect='auto',
                   cmap='afmhot_r', interpolation=None)
    ax.set_ylabel('Raw')
    # ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005).set_label('Value')


def _hog_blocks(fs: FeatureSequence):
    fs_data = fs.np_sequence.T
    return fs_data.T.reshape((fs_data.shape[1], -1, 12))


def hog_blocks_val(fs: FeatureSequence, ax: Axes):
    s = _hog_blocks(fs)
    s = s.max(axis=2).T
    im = ax.imshow(s, extent=(0, fs.duration, .5, 3.5), aspect='auto', cmap='afmhot_r', interpolation=None)
    ax.set_ylabel('Blocks')
    ax.set_yticks([1, 2, 3])
    # ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005).ax.set_ylabel('Magnitude')


def hog_blocks_dir(fs: FeatureSequence, ax: Axes):
    s = _hog_blocks(fs)
    s = s.argmax(axis=2).T
    im = ax.imshow(s, extent=(0, fs.duration, .5, 3.5), aspect='auto', cmap='hsv', interpolation='bicubic')
    ax.set_ylabel('Blocks')
    ax.set_yticks([1, 2, 3])
    # ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005).ax.set_ylabel('Bin')


def _hog_cells(fs: FeatureSequence):
    from engine.features.hog import create_hwr_hog_descriptor, interpret_hog_data, HOGDescriptorProperties
    fs_data = fs.np_sequence.T
    dp = HOGDescriptorProperties(create_hwr_hog_descriptor(256))
    s = interpret_hog_data(fs_data, dp)
    s = s[:, ::2]  # Remove second cell column of each window
    return s


def hog_cells_val(fs: FeatureSequence, ax: Axes):
    s = _hog_cells(fs)
    s = s.max(axis=2)
    im = ax.imshow(s, extent=(0, fs.duration, .5, s.shape[0] + .5), aspect='auto', cmap='afmhot_r',
                   interpolation='bicubic')
    ax.set_ylabel('Cells')
    # ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005).ax.set_ylabel('Magnitude')


def hog_cells_dir(fs: FeatureSequence, ax: Axes):
    s = _hog_cells(fs)
    s = s.argmax(axis=2)
    im = ax.imshow(s, extent=(0, fs.duration, .5, s.shape[0] + .5), vmax=12, aspect='auto', cmap='hsv',
                   interpolation='bicubic')
    ax.set_ylabel('Cells')
    # ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005).ax.set_ylabel('Bin')


def hog_cells_combined(fs: FeatureSequence, ax: Axes):
    s = _hog_cells(fs)
    modulo = 12
    bins = s.argmax(axis=2) % modulo
    vals = s.max(axis=2)
    vals *= 1 / vals.max()
    hsv = np.stack([
        bins / modulo,
        np.ones(s.shape[:2]),
        vals,
        ], axis=2)
    img = hsv2rgb(hsv)
    im = ax.imshow(img, extent=(0, fs.duration, .5, s.shape[0] + .5), aspect='auto', interpolation='bicubic',
                   cmap='Greys_r')
    ax.set_ylabel('Cells')
    # ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005, ticks=[]).ax.set_ylabel('Both')


def hog_cells_combined_unsigned(fs: FeatureSequence, ax: Axes):
    s = _hog_cells(fs)
    modulo = 6
    bins = s.argmax(axis=2) % modulo
    vals = s.max(axis=2)
    vals *= 1 / vals.max()
    hsv = np.stack([
        bins / modulo,
        np.ones(s.shape[:2]),
        vals,
        ], axis=2)
    img = hsv2rgb(hsv)
    im = ax.imshow(img, extent=(0, fs.duration, .5, s.shape[0] + .5), aspect='auto', interpolation='bicubic',
                   cmap='Greys_r')
    ax.set_ylabel('Cells')
    ax.set_xlabel('Time [s]')
    plt.colorbar(im, ax=ax, pad=.005, ticks=[]).ax.set_ylabel('Unsigned')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Preview the contents of a HOG feature file in HTK format\n\n'
                    'Gilles Waeber, I 2021'
    )
    parser.add_argument('files', nargs='+', help='feature files')
    args = parser.parse_args()
    run(**vars(args))
