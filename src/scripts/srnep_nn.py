from engine.export import gpu_lazy_allocation
from scripts.models.cnn1d_model import test_cnn_1d_model
from scripts.models.cnn_model import test_cnn_model
from scripts.models.config import img_height, learning_rate, save_every, hog_num_bins, \
    IMG_PEAK_XPPS_1, IMG_PEAK_XPPS_2, IMG_PEAK_XPPS_3
from scripts.models.datasets import srnep_c6, srnep_c2
from scripts.models.lstm_model import test_lstm2_model_fixed, test_lstm_model_fixed

max_epochs = 200

def run():
    gpu_lazy_allocation()

    for dataset, base in [srnep_c2(), srnep_c6()]:
        data = {
            f'raw': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PEAK_XPPS_1).img_features(),
            f'raw{IMG_PEAK_XPPS_2}': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PEAK_XPPS_2).img_features(),
            f'hog{IMG_PEAK_XPPS_2}': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PEAK_XPPS_2).hog_hwr_features(),
            f'raw{IMG_PEAK_XPPS_3}': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PEAK_XPPS_3).img_features(),
            f'hog{IMG_PEAK_XPPS_3}': base.create_spectrogram(height=img_height, x_pixels_per_sec=IMG_PEAK_XPPS_3).hog_hwr_features(),
        }

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
            lstm_params = dict(
                **common_params,
                lstm_neurons=256,
                no_pad=True,
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


if __name__ == '__main__':
    run()
