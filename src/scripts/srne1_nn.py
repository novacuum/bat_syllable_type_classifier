from engine.export import gpu_lazy_allocation
from scripts.models.cnn1d_model import test_cnn_1d_model
from scripts.models.cnn_model import test_cnn_model
from scripts.models.config import img_height, IMG_PAD_XPPS, IMG_PAD_MAX_WIDTH, learning_rate, save_every, \
    lstm_neurons, padding, IMG_STRETCHED_WIDTH, hog_num_bins
from scripts.models.datasets import srne1_c2, srne1_c5
from scripts.models.lstm_model import test_lstm2_model_fixed, test_lstm_model_fixed

"""Experiments on the SRNE-1 dataset using neural networks

Author: Gilles Waeber, VII 2019"""

max_epochs = 200


def run():
    gpu_lazy_allocation()

    for dataset, base in [srne1_c2(), srne1_c5()]:

        var_data = {
            'raw': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PAD_XPPS).img_features(),
            'hog': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PAD_XPPS).hog_hwr_features()
        }
        stretched_data = {
            'hog': base.create_spectrogram(height=img_height, width=IMG_STRETCHED_WIDTH).hog_hwr_features(),
            'raw': base.create_spectrogram(height=img_height, width=IMG_STRETCHED_WIDTH).img_features(),
        }

        pad_length = {
            'raw': IMG_PAD_MAX_WIDTH,
            'hog': IMG_PAD_MAX_WIDTH // 2
        }

        # Padded
        for f, t in var_data.items():
            common_params = dict(
                dataset=dataset,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                training=t,
                validation=None,  # K-Fold
                testing=None,  # K-Fold
                save_every=save_every,
                prepare_args=dict(
                    padding=padding,
                    padding_length=pad_length[f],
                ),
                model_suffix=f'_{f}_{padding}pad_e{max_epochs}'
            )
            lstm_params = dict(
                **common_params,
                lstm_neurons=lstm_neurons,
            )
            lstm_params['model_suffix'] += f'_n{lstm_params["lstm_neurons"]}'
            test_lstm_model_fixed(**lstm_params)
            test_lstm2_model_fixed(**lstm_params)
            test_cnn_model(**common_params)
            test_cnn_1d_model(**common_params)
            if 'hog' in f:
                params_3d = dict(**common_params)
                params_3d['prepare_args']['new_dim_size'] = hog_num_bins
                params_3d['model_suffix'] += '_3d'
                test_cnn_model(**params_3d)

        # Stretched
        for f, t in stretched_data.items():
            params = dict(
                dataset=dataset,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                training=t,
                validation=None,  # K-Fold
                testing=None,  # K-Fold
                save_every=save_every,
                model_suffix=f'_{f}_e{max_epochs}'
            )
            test_cnn_model(**params)
            test_cnn_1d_model(**params)
            if 'hog' in f:
                params_3d = dict(
                    **params,
                    prepare_args={'new_dim_size': hog_num_bins},
                )
                params_3d['model_suffix'] += '_3d'
                test_cnn_model(**params_3d)


if __name__ == '__main__':
    run()
