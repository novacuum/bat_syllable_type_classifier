from multiprocessing.pool import ThreadPool

from engine.utils import irange
from scripts.models.config import img_height, IMG_PAD_XPPS
from scripts.models.datasets import srne1_c2, srne1_c5
from scripts.models.hmm_full_test import test_hmm_model_k_fold


"""Experiments on the SRNE-1 dataset using Hidden Markov Models

Author: Gilles Waeber, VII 2019"""


def run():
    with ThreadPool() as p:
        for dataset, base in [srne1_c2(), srne1_c5()]:

            var_data = {
                'raw': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PAD_XPPS).img_features(),
                'hog': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PAD_XPPS).hog_hwr_features(),
                'raw200': base.create_spectrogram(height=img_height, x_pixels_per_sec=200).img_features(),
                'hog200': base.create_spectrogram(height=img_height, x_pixels_per_sec=200).hog_hwr_features()
            }

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
