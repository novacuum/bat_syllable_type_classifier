from engine.export import gpu_lazy_allocation
from scripts.models.cnn_model import test_cnn_model
from scripts.models.config import img_height, learning_rate, save_every
from scripts.models.datasets import srnep_c6

"""Experiments on the SRNE-P dataset using Neural Networks and varying spectrogram time sampling rates

Author: Gilles Waeber, VII 2019"""

max_epochs = 200
xppss = [400, 800, 1600]


def run():
    gpu_lazy_allocation()

    for dataset, base in [srnep_c6()]:
        data = dict(
            (f'raw{xpps}', base.create_spectrogram(height=img_height, x_pixels_per_sec=xpps).img_features())
            for xpps in xppss
        )

        for f, t in data.items():
            common_params = dict(
                dataset=dataset,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                training=t,
                validation=None,  # K-Fold
                testing=None,  # K-Fold
                save_every=save_every,
                prepare_args=dict(
                    normalize_samples=True,
                ),
                model_suffix=f'_{f}_e{max_epochs}'
            )
            test_cnn_model(**common_params)


if __name__ == '__main__':
    run()
