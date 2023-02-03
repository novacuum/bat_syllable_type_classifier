from struct import pack, unpack
from typing import Union, List, Optional

import numpy as np

from engine.helpers import TmpFile

"""Management of HTK feature sequence files

Author: Gilles Waeber, 2018"""


def _add_third_dim(data):
    d1, d2 = data.shape
    return data.reshape(1, d1, d2)


def convert_hz_100ns(x: float, /):
    """Convert between a sampling rate given in Hertz and a sampling rate given in units of 100ns (both ways)"""
    return round(1e7 / x)


class FeatureSequence:
    np_sequence: np.ndarray
    x_per_sec: Optional[float]
    """
    A sequence of feature vectors
    With the capability of exporting to/importing from HTK files

    Python struct.pack format: (https://docs.python.org/2/library/struct.html#format-characters)
    <i  signed integer  little-endian  4 bytes
    <h  signed integer  little-endian  2 bytes (half)
    <f  float IEEE 754  little-endian  4 bytes (single-precision)

    HTK file structure: (from the HTK book 3.4.1 §5.10.1)
        HTK Header (12 bytes)
        - number of samples in file     4-byte integer
        - sample period in 100ns units  4-byte integer
        - bytes per sample              2-byte integer
        - sample kind code              2-byte integer
            use code 9 (USER) for custom data
        Followed by a contiguous sequence of samples vectors
        - A sample is either 2-byte integers, when using compression, or 4-byte floats, otherwise
        - Could also contain a CRC or other stuff with a different sample kind code (not handled)
    """

    def __init__(self, data: Union[List[List[float]], np.ndarray], x_per_sec: Optional[float] = None):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.np_sequence = data
        self.x_per_sec = x_per_sec

    def num_feature_vectors(self):
        return self.np_sequence.shape[0]

    def num_features(self):
        return self.np_sequence.shape[1]

    @property
    def duration(self):
        """duration in seconds"""
        if self.x_per_sec is None:
            raise ValueError('Sampling rate not defined!')
        return self.num_feature_vectors() / self.x_per_sec

    @staticmethod
    def from_htk(htk_filename, x_per_sec: Optional[float] = None) -> 'FeatureSequence':
        with open(htk_filename, "rb") as file:
            num_vectors = unpack('<i', file.read(4))[0]
            assert num_vectors > 0, "Invalid vector count"
            htk_sample_rate = unpack('<i', file.read(4))[0]
            if htk_sample_rate == 1:
                pass
            else:
                htk_x_per_sec = convert_hz_100ns(htk_sample_rate)
                if x_per_sec is not None:
                    assert convert_hz_100ns(htk_sample_rate) - 1 <= x_per_sec <= convert_hz_100ns(htk_sample_rate) + 1, \
                        f"Unexpected sample rate of 1/{htk_sample_rate * 100}ns for {x_per_sec} items/s"
                else:
                    x_per_sec = htk_x_per_sec
            sample_size = unpack('<h', file.read(2))[0]
            assert sample_size > 0 and sample_size % 4 == 0, "Invalid sample size"
            num_features = sample_size // 4  # sample size / 4 bytes
            sample_type = unpack('<h', file.read(2))[0]
            assert sample_type == 9, "Unknown sample format"

            return FeatureSequence(np
                                   .fromfile(file, dtype=np.dtype('<f4'), count=num_vectors * num_features)
                                   .reshape(num_vectors, num_features), x_per_sec)

    def to_htk(self, htk_filename):
        sample_size = self.num_features() * 4  # sample size features * 4 bytes

        with TmpFile(htk_filename) as out, open(out, "wb") as file:
            # HTK header
            file.write(pack('<i', self.num_feature_vectors()))  # number of samples
            if self.x_per_sec is not None:
                file.write(pack('<i', convert_hz_100ns(self.x_per_sec)))  # HTK_SAMPLE_RATE
            else:
                file.write(pack('<i', 1))  # dummy HTK_SAMPLE_RATE
            file.write(pack('<h', sample_size))  # sample size = num_features * 4 bytes
            file.write(pack('<h', 9))  # user defined sample kind = 9

            # Contents
            self.np_sequence.astype('<f4', copy=False).tofile(file)


def normalize_one(x: np.ndarray, out_min=-1, out_max=+1) -> np.ndarray:
    x_diff = x.max() - x.min()
    out_diff = out_max - out_min
    return (x - x.min()) * (out_diff / x_diff) + out_min


def normalize_all(x: np.ndarray, out_min=-1, out_max=+1) -> np.ndarray:
    """Normalize multiple samples, sample-wise"""
    broadcast_shape = [-1] + [1] * (len(x.shape) - 1)
    op_axes = tuple(range(1, len(x.shape)))
    x_min = x.min(axis=op_axes).reshape(broadcast_shape)
    x_max = x.max(axis=op_axes).reshape(broadcast_shape)
    x_diff = x_max - x_min
    out_diff = out_max - out_min
    return (x - x_min) * (out_diff / x_diff) + out_min


def merge_to_x(
    features: List[FeatureSequence], *,
    triplicate=False,
    add_dim=True,
    new_dim_size=1,
    kiu_preprocess=False,
    kiu_preprocess_args=None,
    variable_length=False,
    padding=None,
    padding_length=None,
    normalize_samples=True
) -> np.ndarray:
    """Merge feature sequences to an x array for Keras

    Params:
    - triplicate: repeat the gray channel over three channels
    - add_dim: add a fourth dimension to the samples (for 2D convolution)
    - new_dim_size: (with add_dim) size of the fourth dimension
    - kiu_preprocess: use keras imagenet_utils preprocess_input function
    - kiu_preprocess_args: kwargs for the preprocess_input function
    - variable_length: consider the samples as being of variable length (x will be an array of ndarray)
    - padding: pad the samples, 'pre' or 'post'
    - padding_length: new length
    - normalize_sample: normalize values between -1 and 1 for each sample
    """
    if kiu_preprocess_args is None:
        kiu_preprocess_args = {}

    assert len(features), "No features"

    if variable_length:
        assert padding is None, "Do not use variable_length and padding together"
        assert not triplicate and not kiu_preprocess, "Not supported"
        data = [f.np_sequence for f in features]
        if add_dim:
            data = [_add_third_dim(d) for d in data]
        if normalize_samples:
            data = [normalize_one(d) for d in data]
        else:
            assert not normalize_samples, "Not supported"

        import tensorflow as tf
        data = tf.ragged.constant(data, data[0].dtype)
    else:
        if padding is not None:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            assert padding in ('pre', 'post'), "Padding is either pre or post"
            data = [f.np_sequence for f in features]
            assert all(d.shape[0] <= padding_length for d in data), "Some samples are too long"
            # Normalization is done before padding
            if normalize_samples:
                data = [normalize_one(d) for d in data]
            data = pad_sequences(data, padding=padding, truncating=padding, maxlen=padding_length, dtype=data[0].dtype)
        else:
            data = np.stack([f.np_sequence for f in features], axis=0)
            if normalize_samples:
                data = normalize_all(data)

        if add_dim:
            d0, d1, d2 = data.shape
            data = data.reshape([d0, d1, d2 // new_dim_size, new_dim_size])  # A fourth dimension is needed here

        if triplicate:  # Copy the gray channel
            data = np.repeat(data, [3], axis=3)
        if kiu_preprocess:
            from tensorflow.keras.applications.imagenet_utils import preprocess_input
            if kiu_preprocess_args is None:
                kiu_preprocess_args = {}
            data = preprocess_input(data, **kiu_preprocess_args)

    return data
