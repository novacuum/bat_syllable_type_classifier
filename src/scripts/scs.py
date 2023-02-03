from engine.audio import AudioFileList
from engine.export import gpu_lazy_allocation
from engine.processing.audio.splitseq import create_short_identifier
from utils.experiment import run_basic_models, call_variant_function, load_audio_and_db, create_dataset_by_kfold_task, \
    create_basic_features_and_kfold

max_epochs = 100


# configs = [
#     {'part_length': .05, 'strides': .0125, 'label_min_cover_length': .6}
#     , {'part_length': .05, 'strides': .025, 'label_min_cover_length': .6}
#     , {'part_length': .06, 'strides': .015, 'label_min_cover_length': .8}
#     , {'part_length': .06, 'strides': .015, 'label_min_cover_length': .6}
#     , {'part_length': .2, 'strides': .05, 'label_min_cover_length': .8}
#     , {'part_length': .2, 'strides': .05, 'label_min_cover_length': .6}
# ]

# configs = [
#     {'part_length': .05, 'strides': .0125, 'label_min_cover_length': .8}
#     , {'part_length': .05, 'strides': .0125, 'label_min_cover_length': .6}
#     , {'part_length': .02, 'strides': .005, 'label_min_cover_length': .8}
#     , {'part_length': .02, 'strides': .005, 'label_min_cover_length': .6}
# ]

r3_config = [
    {'part_length': .03, 'strides': .005, 'label_min_cover_length': .8}
    , {'part_length': .03, 'strides': .005, 'label_min_cover_length': .6}
    , {'part_length': .03, 'strides': .01, 'label_min_cover_length': .8}
    , {'part_length': .03, 'strides': .01, 'label_min_cover_length': .6}
]


def run():
    # gpu_lazy_allocation()
    call_variant_function(__name__, load_audio_and_db('simple_call_seq'))


def run_variant_r3(ppl: AudioFileList, config_index):
    config = r3_config[config_index]
    ppl = ppl.split_into_parts(**config)

    prefix = '_' + create_short_identifier(**config) + '_'

    feature_data = create_basic_features_and_kfold(ppl.create_spectrogram(sampling_rate=500000, x_pixels_per_sec=4000, height=512, window='Ham'))
    run_basic_models(feature_data, create_dataset_by_kfold_task(feature_data['raw'].task, 'scs_r3', 'audio'), prefix, max_epochs)


if __name__ == '__main__':
    run()
