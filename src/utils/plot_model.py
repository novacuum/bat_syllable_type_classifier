import json
from argparse import ArgumentParser
from collections import defaultdict
from os import path
from pathlib import Path

import matplotlib.pyplot as plt

from engine.helpers import read_file
from engine.settings import MODELS_FOLDER


def training_plot(input_file, *, fig=None, axes=None, figsize=(8, 2.5), separate_loss_axes=False):
    data = json.loads(read_file(input_file))
    if axes is None:
        fig, axes = plt.subplots(ncols=2, figsize=figsize)
    ax = axes[0]
    ax.plot(range(1, len(data) + 1), list(h['acc'] for h in data), label='training')
    ax.plot(range(1, len(data) + 1), list(h['val_acc'] for h in data), label='validation')
    ax.set_ylabel('accuracy'), ax.set_xlabel('epochs')
    ax.legend(), ax.grid()
    ax = axes[1]
    ax.plot(range(1, len(data) + 1), list(h['loss'] for h in data), color='C0')
    axv = ax.twinx() if separate_loss_axes else ax
    axv.plot(range(1, len(data) + 1), list(h['val_loss'] for h in data), color='C1')
    ax.set_ylabel('loss (categorical cross-entropy)'), ax.set_xlabel('epochs')
    if separate_loss_axes:
        axv.set_ylabel('validation loss')
    ax.grid()
    return fig


def find_stats_file(input):
    in_path = Path(input)
    if in_path.suffix == 'json' and in_path.is_file():
        return input
    elif in_path.is_dir():
        cnd = list(in_path.glob('epoch_*.h5.stats.json'))
        if len(cnd):
            return sorted(cnd, reverse=True)[0]
        else:
            cnd = list(in_path.glob('*'))
            assert len(cnd), f'Found nothing in {input}'
            assert len(cnd) == 1, f'Multiple possibilities in {input}'
            return find_stats_file(cnd[0])
    elif path.isdir(f'{MODELS_FOLDER}/{input}'):
        return find_stats_file(f'{MODELS_FOLDER}/{input}')
    elif path.isdir(f'{MODELS_FOLDER}/nn_{input}'):
        return find_stats_file(f'{MODELS_FOLDER}/nn_{input}')
    else:
        raise RuntimeError(f'Cannot find {input}')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Plot training stats for a model\n\n'
                    'Gilles Waeber, VII 2019'
    )
    parser.add_argument('models', metavar='model', nargs='+', help='model folder or name')
    parser.add_argument('-c', '--combine', action='store_true', help='combine all plots into a single plot')
    parser.add_argument('-o', '--output', default=None, help='store the plot in a file (assumes -c)')
    parser.add_argument('-f', '--format', default=None,
                        help='output format, when storing in a file (uses filename when not specified)')
    parser.add_argument('-s', '--separate-loss-axes', action='store_true', help='use two y axis for losses')
    args = parser.parse_args()
    # print(args)
    if args.output:
        args.combine = True
    fig, axes = None, defaultdict(lambda: None)
    if args.combine:
        fig, axes = plt.subplots(nrows=len(args.models), ncols=2, figsize=(8, 2.5 * len(args.models)))
        if len(args.models) < 2:
            axes = [axes]
    for r, a in enumerate(args.models):
        fig = training_plot(find_stats_file(a), fig=fig, axes=axes[r], separate_loss_axes=args.separate_loss_axes)
        if args.combine:
            axes[r][0].text(0, 1.05, a, ha='left', transform=axes[r][0].transAxes)
        else:
            fig.suptitle(a)
            fig.show()
    if args.combine:
        fig.subplots_adjust(hspace=.5, wspace=.3)
        if args.output is None:
            fig.show()
    if args.output is not None:
        fig.savefig(args.output, bbox_inches='tight', format=args.format)
    else:
        plt.show()
