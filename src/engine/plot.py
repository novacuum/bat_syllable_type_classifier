from collections import defaultdict
from dataclasses import dataclass
import matplotlib.pyplot as plt
import string
import numpy as np

"""Utilities for creating graphs

Author: Gilles Waeber, VII 2019"""

alphabets = [l for l in zip(string.ascii_uppercase, string.ascii_lowercase, "αβγδεζηθικλμνξοπρςτυφχψω")]
labels = ['B2', 'B3', 'B4', 'VS', 'VSV', 'UPS', 'none']
symbols = {label: alphabets[i] for i, label in enumerate(labels)}

# colors = defaultdict(lambda: (.2, .2, .2), [
#     ('cc05', (.5, 0, .5)), ('5', (.5, 0, .5)), ('ind05', (.5, 0, .5)),
#     ('cc19', (0, 0, .5)), ('19', (0, 0, .5)), ('ind19', (0, 0, .5)),
#     ('cc23', (0, .5, 0)), ('23', (0, .5, 0)), ('ind23', (0, .5, 0)),
#     ('cc24', (.5, .5, 0)), ('24', (.5, .5, 0)), ('ind24', (.5, .5, 0)),
#     ('cc25', (.7, 0, 0)), ('25', (.7, 0, 0)), ('ind25', (.7, 0, 0)),
#     ('cc30', (0, .5, .5)), ('30', (0, .5, .5)), ('ind30', (0, .5, .5)),
#
#     ('cc', (.7, 0, 0)),
#     ('c', (.5, 0, .5)),
#     ('w', (0, .5, 0)),
# ])

label_color_map = plt.cm.get_cmap('viridis', len(labels)) #cubehelix
colors = {label: label_color_map(i) for i, label in enumerate(symbols.keys())}


def dist_matrix(size=5):
    import numpy as np
    x, y = np.ogrid[0:size, 0:size]
    return np.hypot(x - size // 2, y - size // 2)


def add_value_labels(ax, spacing=10, horizontal=False):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

    Source:
        https://stackoverflow.com/questions/28931224/adding-value-labels-on-a-matplotlib-bar-chart
    """

    @dataclass(frozen=True)
    class Bar:
        x_pos: float
        y_pos: float
        value: str

    # Get X and Y placement of label from rect.
    if horizontal:
        patches = defaultdict(list)
        for r in ax.patches:
            patches[r.get_y()].append(r)
        bars = [Bar(
            max(r.get_x() + r.get_width() for r in rs),
            y + rs[0].get_height() / 2,
            '+'.join(str(r.get_width()) for r in rs)
        ) for y, rs in patches.items()]
    else:
        patches = defaultdict(list)
        for r in ax.patches:
            patches[r.get_x()].append(r)
        bars = [Bar(
            x + rs[0].get_weight() / 2,
            max(r.get_y() + r.get_height() for r in rs),
            '+'.join(str(r.get_height()) for r in rs)
        ) for x, rs in patches.items()]

    # For each bar: Place a label
    for bar in bars:
        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Alignment for positive values
        va = 'center' if horizontal else 'bottom'
        ha = 'center' if not horizontal else 'left'

        # If value of bar is negative: Place label below bar
        # if value < 0:
        #    # Invert space to place label below
        #    space *= -1
        #    # Vertically align label at top
        #    if not horizontal:
        #        va = 'top'

        shift = (space, 0) if horizontal else (0, space)

        # Create annotation
        ax.annotate(
            bar.value,  # Use `label` as label
            (bar.x_pos, bar.y_pos),  # Place label at end of the bar
            xytext=shift,  # Shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            va=va, ha=ha)  # Align label differently for positive and negative values.


def place_texts(points, x_lim, y_lim, iterations=2):
    import numpy as np
    from scipy import signal
    from engine.helpers import BoundingBox

    DIM = 750
    DOT_W, DOT_H, DOT_M = 4, 3, .01
    dot_block = np.ones((1 + 2 * DOT_W, 1 + 2 * DOT_H))
    dot_patch = signal.convolve2d(dot_block, dot_block) * DOT_M
    dpx, dpy = dot_patch.shape[0] // 2, dot_patch.shape[1] // 2
    TEXT_W, TEXT_H, TEXT_P = 5, 4, 6
    TEXT_D = 100
    text_distance = (TEXT_D * 1.414 - dist_matrix(2 * TEXT_D)) / (TEXT_D * 1.414)
    text_block = np.ones((1 + 2 * TEXT_W, 1 + 2 * TEXT_H))
    text_patch = signal.convolve2d(text_block, text_block)
    tpx, tpy = text_patch.shape[0] // 2, text_patch.shape[1] // 2
    bb = BoundingBox((DIM - TEXT_D, TEXT_D), (TEXT_D, DIM - TEXT_D), y_lim, x_lim)
    fill = np.zeros((DIM, DIM))
    fill[0:TEXT_D, :] = 1000
    fill[DIM - TEXT_D:DIM, :] = 1000
    fill[:, 0:TEXT_D] = 1000
    fill[:, DIM - TEXT_D:DIM] = 1000

    for pos in points:
        x, y = int(bb.x(pos[0])), int(bb.y(pos[1]))
        fill[x - dpx:x + dpx + 1, y - dpy:y + dpy + 1] += dot_patch

    fill = signal.convolve2d(fill, text_block)

    text_pos = [(int(bb.x(pos[0])), int(bb.y(pos[1]))) for pos in points]
    for x, y in text_pos:
        fill[x - tpx:x + tpx + 1, y - tpy:y + tpy + 1] += text_patch
    # plt_im_show(fill[TEXT_D + tpx:DIM - TEXT_D - tpx, TEXT_D + tpy:DIM - TEXT_D - tpy].transpose(), cmap='seismic', plt_show=False)

    for _ in range(iterations):
        new_pos = []
        for x, y in text_pos:
            # print(f'Before: {(x, y)}')
            fill[x - tpx:x + tpx + 1, y - tpy:y + tpy + 1] -= text_patch
            cfill = fill.copy()
            cfill[x - TEXT_D:x + TEXT_D, y - TEXT_D:y + TEXT_D] -= TEXT_P * text_distance
            x, y = np.unravel_index(cfill.argmin(), cfill.shape)
            # print(f'After: {(x, y)}')
            new_pos.append((x, y))
            fill[x - tpx:x + tpx + 1, y - tpy:y + tpy + 1] += text_patch
        text_pos = new_pos

    return [(bb.x_rev(x), bb.y_rev(y)) for x, y in text_pos]


def tsne_plot(ff, ax, labels=True):
    mp = ff.create_nn_model("dummy", {}).task.mp
    x, y, _ = ff.get_xyw(mp)
    return tsne_plot_xym(x, y, mp, ax, labels)


def tsne_plot_xym(x, y, mp, ax, labels=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy import signal
    from sklearn import manifold
    from engine.features.feature_sequence import merge_to_x, FeatureSequence
    from engine.helpers import BoundingBox
    from tqdm import tqdm

    ay = y.argmax(axis=1)
    x = x.reshape((x.shape[0], -1))
    y_text = [mp.num_label[y] for y in ay]
    sym = []
    texts = list(sorted(set(y_text)))

    label_count = defaultdict(lambda: 0)
    label_prefixes = defaultdict(list)
    for t in y_text:
        label_count[t] += 1
        if labels:
            num = label_count[t]
            if num == 1 or not num % 100:
                label_prefixes[t].append(symbols[t][num // 100])
            sym.append(f'{symbols[t][num // 100]}{num % 100}')

    x_new = manifold.TSNE(n_components=2, random_state=2).fit_transform(x)
    ax.scatter(x_new[:, 0], x_new[:, 1], c=[colors[y] for y in y_text])

    if labels:
        DIM = 750
        DOT_W, DOT_H, DOT_M = 4, 3, .01
        dot_block = np.ones((1 + 2 * DOT_W, 1 + 2 * DOT_H))
        dot_patch = signal.convolve2d(dot_block, dot_block) * DOT_M
        dpx, dpy = dot_patch.shape[0] // 2, dot_patch.shape[1] // 2
        TEXT_W, TEXT_H, TEXT_P = 5, 4, 6
        TEXT_D = 100
        text_distance = (TEXT_D * 1.414 - dist_matrix(2 * TEXT_D)) / (TEXT_D * 1.414)
        text_block = np.ones((1 + 2 * TEXT_W, 1 + 2 * TEXT_H))
        text_patch = signal.convolve2d(text_block, text_block)
        tpx, tpy = text_patch.shape[0] // 2, text_patch.shape[1] // 2
        bb = BoundingBox((DIM - TEXT_D, TEXT_D), (TEXT_D, DIM - TEXT_D), ax.get_ylim(), ax.get_xlim())
        fill = np.zeros((DIM, DIM))
        fill[0:TEXT_D, :] = 1000
        fill[DIM - TEXT_D:DIM, :] = 1000
        fill[:, 0:TEXT_D] = 1000
        fill[:, DIM - TEXT_D:DIM] = 1000

        for pos in x_new:
            x, y = int(bb.x(pos[0])), int(bb.y(pos[1]))
            fill[x - dpx:x + dpx + 1, y - dpy:y + dpy + 1] += dot_patch

        fill = signal.convolve2d(fill, text_block)

        text_pos = [(int(bb.x(pos[0])), int(bb.y(pos[1]))) for pos in x_new]
        for x, y in text_pos:
            fill[x - tpx:x + tpx + 1, y - tpy:y + tpy + 1] += text_patch

        for _ in range(3):
            new_pos = []
            for x, y in tqdm(text_pos):
                # print(f'Before: {(x, y)}')
                fill[x - tpx:x + tpx + 1, y - tpy:y + tpy + 1] -= text_patch
                cfill = fill.copy()
                cfill[x - TEXT_D:x + TEXT_D, y - TEXT_D:y + TEXT_D] -= TEXT_P * text_distance
                x, y = np.unravel_index(cfill.argmin(), cfill.shape)
                # print(f'After: {(x, y)}')
                new_pos.append((x, y))
                fill[x - tpx:x + tpx + 1, y - tpy:y + tpy + 1] += text_patch
            text_pos = new_pos

        for pos, s, (x, y), t in zip(x_new, sym, text_pos, y_text):
            ax.annotate(s, xy=pos, xytext=(bb.x_rev(x), bb.y_rev(y)), ha='center', va='center', fontSize=6,
                        color=colors[t],
                        arrowprops={'arrowstyle': '-', 'color': colors[t]})

    ax.legend(handles=[Line2D([0], [0], marker='o', color=colors[l],
                              label=f'{l} ({label_count[l]} items' +
                                    (f', prefix: {"".join(label_prefixes[l])}' if labels else '') + ')'
                              ) for l in texts])
    ax.set_ylabel('t-SNE axis 2'), ax.set_xlabel('t-SNE axis 1')
    return ax


def polar2cart(r, theta, phi):
    import numpy as np
    return np.column_stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ])


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
