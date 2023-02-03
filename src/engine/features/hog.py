from __future__ import annotations

import math
from typing import Tuple, TYPE_CHECKING

import numpy as np

from engine.features.feature_sequence import FeatureSequence

if TYPE_CHECKING:
    import cv2


"""HOG Feature extration

Author: Gilles Waeber, VI 2019"""


def extract_hog_features(src_file, dest_file, x_per_sec):
    import cv2
    img: np.ndarray = cv2.imread(src_file, cv2.IMREAD_GRAYSCALE)
    rows, cols = img.shape
    window_size = 16  # From HWRecog
    stride = 2  # From HWRecog
    n_windows = int(math.ceil((cols - window_size) / stride) + 1)  # From HWRecog

    # From HWRecog: Padding to make width of the image a multiple of window size
    padding = (n_windows - 1) * stride + window_size
    img_padded: np.ndarray = cv2.copyMakeBorder(src=img, top=0, bottom=0, left=0, right=padding,
                                                borderType=cv2.BORDER_CONSTANT,
                                                value=255)
    img_blurred: np.ndarray = cv2.GaussianBlur(src=img_padded, ksize=(13, 13), sigmaX=3,
                                               sigmaY=3)  # Settings from HWRecog

    hog_descriptor = create_hwr_hog_descriptor(rows)
    features = []
    for i in range(n_windows):
        x1, x2 = stride * i, stride * i + window_size
        window = img_blurred[:, x1:x2]
        desc = hog_descriptor.compute(window).reshape((-1))
        features.append(desc)
    fs = FeatureSequence(np.stack(features, axis=0), x_per_sec)
    fs.to_htk(dest_file)


def create_hwr_hog_descriptor(rows):
    """Create the HOG descriptor used in HWRecog"""
    window_size = 16  # From HWRecog
    return create_hog_descriptor(
        win_size=(window_size, (rows // 2) * 2),
        block_size=(window_size, rows // 2),
        block_stride=(window_size, rows // 4),
        cell_size=(window_size // 4, rows // 4),
        n_bins=12,  # default is 9
        gamma_correction=False,
        signed_gradients=True
    )


def create_hog_descriptor(
    *,
    win_size=(64, 128),
    block_size=(16, 16),
    block_stride=(8, 8),
    cell_size=(8, 8),
    n_bins=9,
    deriv_aperture=1,
    win_sigma=-1,
    histogram_norm_type=0,
    l2_hys_threshold=.2,
    gamma_correction=True,
    n_levels=64,
    signed_gradients=False
) -> cv2.HOGDescriptor:
    """Create a HOG descriptor"""
    # https://stackoverflow.com/a/31042147
    import cv2
    return cv2.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        n_bins,
        deriv_aperture,
        win_sigma,
        histogram_norm_type,
        l2_hys_threshold,
        gamma_correction,
        n_levels,
        signed_gradients
    )


class HOGDescriptorProperties:
    gradient_strengths: np.ndarray

    def __init__(
        self,
        descriptor: cv2.HOGDescriptor = None,
        *,
        win_size: Tuple[int, int] = (64, 128),
        block_size: Tuple[int, int] = (16, 16),
        block_stride: Tuple[int, int] = (8, 8),
        cell_size: Tuple[int, int] = (8, 8),
        n_bins=9,
        signed_gradients=False,  # over 360 degrees instead of 180
    ):
        if descriptor is not None:
            win_size = descriptor.winSize
            cell_size = descriptor.cellSize
            n_bins = descriptor.nbins
            block_size = descriptor.blockSize
            block_stride = descriptor.blockStride
            signed_gradients = descriptor.signedGradient

        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.n_bins = n_bins
        self.signed_gradients = signed_gradients

        # dividing 180° into bins, how large (in rad) is one bin?
        self.radians_per_bin = (2 * math.pi if signed_gradients else math.pi) / n_bins

        # compute number of cells
        assert not win_size[0] % cell_size[0] and not win_size[1] % cell_size[1]
        assert not block_size[0] % cell_size[0] and not block_size[1] % cell_size[1]
        self.n_cells_w = win_size[0] // cell_size[0]
        self.n_cells_h = win_size[1] // cell_size[1]
        self.n_cells = self.n_cells_w * self.n_cells_h
        self.block_size_c = block_size[0] // cell_size[0], block_size[1] // cell_size[1]
        self.n_cells_block = self.block_size_c[0] * self.block_size_c[1]

        # compute the number of blocks
        assert not win_size[0] % block_stride[0] and not win_size[1] % block_stride[1]
        self.n_blocks_w = len(range(block_size[0], win_size[0], block_stride[0])) + 1
        self.n_blocks_h = len(range(block_size[1], win_size[1], block_stride[1])) + 1
        self.n_blocks = self.n_blocks_w * self.n_blocks_h


def interpret_hog_data(
    values: np.ndarray,
    d: HOGDescriptorProperties,
    *,
    verbose=0
):
    cols = np.hsplit(values, values.shape[1])
    return np.concatenate([interpret_hog_vector(v, d, verbose=verbose) for v in cols], axis=1)


def interpret_hog_vector(
    values: np.ndarray,
    d: HOGDescriptorProperties,
    *,
    verbose=0
):
    # prepare data structure: bin values / gradient strengths for each cell
    gradient_strengths = np.zeros(shape=(d.n_cells_h, d.n_cells_w, d.n_bins))
    cell_update_counter = np.zeros(shape=(d.n_cells_h, d.n_cells_w))

    if verbose > 0:
        print(f'Cell: {d.cell_size[0]}x{d.cell_size[1]} px')
        print(f'Block: {d.block_size_c[0]}x{d.block_size_c[1]} cells')
        print(f'Window: {d.n_blocks_w}x{d.n_blocks_h} blocks, {d.n_cells_w}x{d.n_cells_h} cells')

    assert len(values) == d.n_blocks * d.n_cells_block * d.n_bins

    val_idx = values.reshape((d.n_blocks_w, d.n_blocks_h, d.block_size_c[0], d.block_size_c[1], d.n_bins), )

    # compute gradient strengths per cell
    for block_x in range(d.n_blocks_w):
        for block_y in range(d.n_blocks_h):
            # Overlapping blocks lead to multiple updates of this sum!
            # we therefore keep track how often a cell was updated, to compute average gradient strengths
            cell_update_counter[block_y:block_y + d.block_size_c[1], block_x:block_x + d.block_size_c[0]] += 1

            for block_cell_x in range(d.block_size_c[0]):
                for block_cell_y in range(d.block_size_c[1]):
                    # compute corresponding cell nr
                    cell_x = block_x + block_cell_x
                    cell_y = block_y + block_cell_y

                    strength = val_idx[block_x, block_y, block_cell_x, block_cell_y, :]
                    gradient_strengths[cell_y, cell_x, :] += strength

    # compute average gradient strengths
    gradient_strengths = gradient_strengths / cell_update_counter.reshape((d.n_cells_h, d.n_cells_w, 1), )

    return gradient_strengths


def visualize_hog_descriptor(
    src_img: np.ndarray,
    values: np.ndarray,
    d: HOGDescriptorProperties,
    *,
    win_size: Tuple[int, int] = (64, 128),
    block_size: Tuple[int, int] = (16, 16),
    block_stride: Tuple[int, int] = (8, 8),
    cell_size: Tuple[int, int] = (8, 8),
    n_bins=9,
    signed_gradients=False,  # over 360 degrees instead of 180
    img_scale=1.0,
    viz_factor=1.0,
    verbose=0
) -> np.ndarray:
    """Visualize HOG features

    Adapted for arbitrary size of feature sets and training images, adapted from C++ code at
    http://www.juergenbrauer.org/old_wiki/doku.php?id=public:hog_descriptor_computation_and_visualization"""
    import cv2

    orig_cols, orig_rows = src_img.shape
    visual_image: np.ndarray = cv2.resize(src=src_img, dsize=(int(orig_rows * img_scale), int(orig_cols * img_scale)))
    visual_image = cv2.cvtColor(visual_image, cv2.COLOR_GRAY2RGB)

    gradient_strengths = interpret_hog_vector(values, d, verbose=verbose)

    # draw cells
    for cell_y in range(d.n_cells_h):
        for cell_x in range(d.n_cells_w):
            draw_x = cell_x * cell_size[0]
            draw_y = cell_y * cell_size[1]

            mx = draw_x + cell_size[0] / 2
            my = draw_y + cell_size[1] / 2

            cv2.rectangle(visual_image,
                          (int(round(draw_x * img_scale)), int(round(draw_y * img_scale))),
                          (int(round((draw_x + cell_size[0]) * img_scale)),
                           int(round((draw_y + cell_size[1]) * img_scale))),
                          (0, 255, 255),
                          1)

            # draw in each cell all 9 gradient strengths
            for b in range(n_bins):
                strength = gradient_strengths[cell_y, cell_x, b]
                angle_rad = (b + .5) * d.radians_per_bin

                dir_x = math.cos(angle_rad)
                dir_y = math.sin(angle_rad)
                max_len = cell_size[0] / 2

                # compute line coordinates
                if signed_gradients:
                    x1, y1 = mx, my
                else:
                    x1 = mx - dir_x * strength * max_len * viz_factor
                    y1 = my - dir_y * strength * max_len * viz_factor
                x2 = mx + dir_x * strength * max_len * viz_factor
                y2 = my + dir_y * strength * max_len * viz_factor

                # draw gradient visualization
                cv2.line(visual_image,
                         (int(round(x1 * img_scale)), int(round(y1 * img_scale))),
                         (int(round(x2 * img_scale)), int(round(y2 * img_scale))),
                         (0, 0, 255),
                         1)

    return visual_image
