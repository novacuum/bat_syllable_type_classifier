from datetime import datetime

from engine.audio import AudioFileList
from engine.settings import LOG_FOLDER
from scripts.models.config import learning_rate, save_every
from scripts.models.lstm_model import test_lstm_model
from utils.experiment import run_basic_models, create_dataset_by_kfold_task, create_basic_features_and_kfold, call_variant_function, load_audio_and_db
from utils.task import create_shortened_identifier_with_value
import tensorflow as tf

max_epochs = 100

padded_config = [
    {'noise_reduction_sensitivity': 0, 'spectrogram': {'x_pixels_per_sec': 2000, 'height': 256}}
    , {'noise_reduction_sensitivity': 0, 'spectrogram': {'x_pixels_per_sec': 5000, 'height': 512}}
    , {'noise_reduction_sensitivity': 6, 'spectrogram': {'x_pixels_per_sec': 2000, 'height': 256}}
    , {'noise_reduction_sensitivity': 6, 'spectrogram': {'x_pixels_per_sec': 5000, 'height': 512}}
]

compressed_config = [
    {'noise_reduction_sensitivity': 0}
    , {'noise_reduction_sensitivity': 6}
    , {'noise_reduction_sensitivity': 12}
    , {'noise_reduction_sensitivity': 24}
]

variable_length_config = [
    {'x_pixels_per_sec': 5000, 'height': 256}
    , {'x_pixels_per_sec': 5000, 'height': 512}
    , {'x_pixels_per_sec': 5000, 'height': 300}
    , {'x_pixels_per_sec': 2000, 'height': 256}
    # , {'x_pixels_per_sec': 4000, 'height': 256}
    # , {'x_pixels_per_sec': 4000, 'height': 512}
    # , {'x_pixels_per_sec': 4000, 'height': 300}
]


def run():
    # tf.debugging.experimental.enable_dump_debug_info(
    #     str(LOG_FOLDER / 'tensorboard' / datetime.now().strftime("%Y%m%d-%H%M%S")), tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1
    # )
    call_variant_function(__name__, load_audio_and_db('simple_call_test'))


def run_variant_variable_length(ppl: AudioFileList, config_index):
    config = variable_length_config[config_index]
    ppl = ppl.extract_label_parts(False).create_spectrogram(sampling_rate=500000, **config, window='Ham')
    var_data = create_basic_features_and_kfold(ppl)
    dataset = create_dataset_by_kfold_task(var_data['raw'].task, 'sct_vl2', 'audio')
    identifier = create_shortened_identifier_with_value(**config)

    for f, t in var_data.items():
        # ignore following setup, creates to small images
        if (config['x_pixels_per_sec'] == 2000 and f == 'hog') or f == 'raw':
            continue

        common_params = dict(
            dataset=dataset,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            training=t,
            validation=None,  # K-Fold
            testing=None,  # K-Fold
            save_every=save_every,
            prepare_args={'variable_length': True},
            model_suffix=f'_{identifier}_{f}_{max_epochs}',
        )

        lstm_params = dict(
            **common_params,
            lstm_neurons=256,
        )

        test_lstm_model(**lstm_params)


def run_variant_compressed(ppl: AudioFileList, config_index):
    config = compressed_config[config_index]
    ppl = ppl.duration_length_filter(0.022).min_count_label_filter(50).extract_label_parts(False)

    if config['noise_reduction_sensitivity'] > 0:
        ppl = ppl.noise_reduce(ppl.task.src_list.create_silent_derivatives(0.3).noise_profile(),
                               config['noise_reduction_sensitivity'])

    prefix = '_' + create_shortened_identifier_with_value('', **config) + '_'
    feature_data = create_basic_features_and_kfold(ppl.create_spectrogram(sampling_rate=500000, width=100, window='Ham'))
    run_basic_models(feature_data, create_dataset_by_kfold_task(feature_data['raw'].task, 'sct_compressed', 'audio'), prefix, max_epochs)


def run_variant_padded(ppl: AudioFileList, config_index):
    config = padded_config[config_index]
    ppl = ppl.extract_label_parts(True, True)

    if config['noise_reduction_sensitivity'] > 0:
        ppl = ppl.noise_reduce(ppl.task.silent_audio_list.noise_profile(), config['noise_reduction_sensitivity'])

    prefix = '_' + create_shortened_identifier_with_value('', **config) + '_'
    feature_data = create_basic_features_and_kfold(
        ppl.create_spectrogram(sampling_rate=500000, **config['spectrogram'], window='Ham')
    )

    # run densNet not on big samples, not enough memory, needs specialized training strategy
    ignore_densnet = config['spectrogram']['x_pixels_per_sec'] == 5000 or config['spectrogram']['height'] == 512
    run_basic_models(feature_data, create_dataset_by_kfold_task(feature_data['raw'].task, 'sct_left_padded', 'audio'), prefix, max_epochs, ignore_densnet)


if __name__ == '__main__':
    run()
