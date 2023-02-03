from multiprocessing.pool import ThreadPool

from engine.utils import irange
from scripts.models.config import img_height, IMG_PEAK_XPPS_1, IMG_PEAK_XPPS_2, IMG_PEAK_XPPS_3
from scripts.models.datasets import srnep_c2, srnep_c6
from scripts.models.hmm_full_test import test_hmm_model_k_fold

"""Experiments on the SRNE-P dataset using Hidden Markov Models

Author: Gilles Waeber, VII 2019"""


def run():
    with ThreadPool() as p:
        for dataset, base in [srnep_c6(), srnep_c2()]:

            var_data = {
              #  f'raw': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PEAK_XPPS_1).img_features(),
              #  f'raw{IMG_PEAK_XPPS_2}': base.create_spectrogram(height=img_height,
              #                                                   x_pixels_per_sec=IMG_PEAK_XPPS_2).img_features(),
              #  f'hog{IMG_PEAK_XPPS_2}': base.create_spectrogram(height=img_height,
              #                                                   x_pixels_per_sec=IMG_PEAK_XPPS_2).hog_hwr_features(),
                f'raw{200}': base.create_spectrogram(height=img_height, x_pixels_per_sec=200).img_features(),
                f'raw{800}N': base.create_spectrogram(height=img_height, x_pixels_per_sec=800, norm=-3).img_features(),
                f'raw{400}N': base.create_spectrogram(height=img_height, x_pixels_per_sec=400, norm=-3).img_features(),
            }
            for data in var_data.values():
                data.run()

            for f, data in var_data.items():
                test_hmm_model_k_fold(
                    dataset=dataset,
                    mixtures=irange(4, 30),
                    states=irange(3, 8),
                    base=data,
                    model_suffix=f'_{f}',
                    pool=p
                )


if __name__ == '__main__':
    run()
