from argparse import ArgumentParser
from collections import defaultdict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.text import Text

from engine.audio import load_audio
from engine.features.feature_sequence import merge_to_x, FeatureSequence
from engine.hacked.t_sne import TSNE
from engine.metadata import metadata_db
from engine.plot import symbols, colors
from scripts.models.datasets import BEFORE_PEAK, AFTER_PEAK


def polar2cart(r, theta, phi):
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])


def minmax(v):
    return min(v), max(v)


def rgb_to_indexed(img):
    index = np.unique(img.reshape((-1, 3)), axis=0)
    color_ids = np.array(range(index.shape[0]))

    # Initialize output array
    result = np.empty((img.shape[0], img.shape[1]), dtype=int)
    result[:] = -1

    # Finally get the matches and accordingly set result locations
    # to their respective color IDs
    R, C, D = np.where((img == index[:, None, None, :]).all(3))
    result[C, D] = color_ids[R]

    return result, [tuple(c) for c in index]


def run(perplexity=30, iterations=1000, random_state=None):
    perplexity = int(perplexity)
    iterations = int(iterations)

    DATASET = 'min100/all.csv'
    IMG_WIDTH = 200
    IMG_HEIGHT = 256
    mdb = metadata_db('manual_peak_1/metadata.json')

    base = load_audio(mdb, DATASET).extract_peaks(0.05, 0.05).create_spectrogram(sampling_rate=500000, x_pixels_per_sec=2000, window='Ham').img_features().run()
    x = merge_to_x([FeatureSequence.from_htk(f.path()) for f in base.files])
    x = x.reshape((x.shape[0], -1))

    # theta = np.arange(30, 160, 5) * (2 * np.pi / 360)
    # phi = np.arange(0, 330, 5) * (2 * np.pi / 360)
    # uv_sphere = np.transpose([np.tile(theta, len(phi)), np.repeat(phi, len(theta))])
    # x = polar2cart(1, uv_sphere[:, 0], uv_sphere[:, 1])
    # map: np.ndarray = plt.imread('/home/gilles/workspace/b/resources/earthmap1k5.png')
    # print(map.shape, map.min(), map.max())
    # map, cmap = rgb_to_indexed(map)
    # print(f'{x.shape} sphere')
    # print(f'{len(cmap)} colors')

    plt.scatter(x[:, 1], x[:, 2])
    plt.show()
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

    # plt_im_show(map)
    # bb = BoundingBox((0, map.shape[1]), (0, map.shape[0]), (0, 2 * math.pi), (0, math.pi))

    slices = []
    TSNE(n_components=2, random_state=random_state, perplexity=perplexity, verbose=True,
         n_iter=iterations).fit_transform(x, save_slices_to=slices)

    # files = sorted(list(Path.cwd().glob('sne???.npy')))
    # print(files[:10])
    # data = np.stack(np.load(f).reshape((-1, 2)) for f in files)
    data = np.stack([s.reshape((-1, 2)) for s in slices])
    y_text = [f.metadata.label for f in base.files]
    last = len(data) - 1
    x_mm = (data[:, :, 0].min(), data[:, :, 0].max())
    x_lim = (data[-1, :, 0].min() - 2, data[-1, :, 0].max() + 2)
    y_mm = (data[:, :, 1].min(), data[:, :, 1].max())
    y_lim = (data[-1, :, 1].min() - 2, data[-1, :, 1].max() + 2)
    data_cl = data.clip([x_lim[0], y_lim[0]], [x_lim[1], y_lim[1]])
    print(x_mm, x_lim)
    print(y_mm, y_lim)
    print(data.shape)

    sym = []
    labels = list(sorted(set(y_text)))
    label_count = defaultdict(lambda: 0)
    label_prefixes = defaultdict(list)
    for t in y_text:
        label_count[t] += 1
        num = label_count[t]
        if num == 1 or not num % 2000:
            label_prefixes[t].append(symbols[t][num // 2000])
        sym.append(f'{symbols[t][num // 2000]}{num % 2000}')

    # create a figure with an axes
    fig, ax = plt.subplots()
    # set the axes limits
    ax.axis([*x_lim, *y_lim])
    # set equal aspect such that the circle is not shown as ellipse
    # ax.set_aspect("equal")
    # create a point in the axes

    points: List[Line2D] = [ax.plot(d1, d2, marker="o", c=colors[y])[0] for (d1, d2), y in zip(data[0], y_text)]
    # points: List[Line2D] = [ax.plot(d1, d2, marker="o")[0] for (d1, d2), y in zip(data[0], y_text)]

    # points: List[Line2D] = [ax.plot(d1, d2, marker="o", c=cmap[map[int(bb.x(u)), int(bb.y(v))]])[0] for (d1, d2), (u, v) in zip(data[0], uv_sphere)]
    # annotations = [ax.annotate(s, xy=point_xy, xytext=text_xy, ha='center', va='center', arrowprops={'arrowstyle': '-'})
    #               for point_xy, text_xy, s in zip(data[0], place_texts(data[0], x_lim, y_lim), sym)]
    text: Text = ax.text(0, 1.05, f'Iteration 0/{last}', ha='left', transform=ax.transAxes)
    artists = points + [text]

    def update(t):
        # t = t * 10
        for i, p in enumerate(points):
            p.set_data((data[t, i, 0],), (data[t, i, 1],))
        text.set_text(f'Iteration {t}/{last}')
        return artists

    # ani = Player(fig, update, mini=0, maxi=len(data), repeat=True)

    ani = FuncAnimation(fig, update, interval=1, blit=False, repeat=True, frames=range(len(data)))
    ani.to_jshtml()
    fig.suptitle(f't-SNE {DATASET}, p={perplexity}')
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='t-SNE animation')
    parser.add_argument('-p', '--perplexity', default=30, help='t-SNE perplexity')
    parser.add_argument('-r', '--random-state', default=None, help='Random state')
    parser.add_argument('-i', '--iterations', default=1000, help='t-SNE iterations')
    args = parser.parse_args()
    run(**vars(args))
