import json

from .audio import AudioLoadTask, AudioMatchingFilesLoadTask, AudioMatchTask
from .audio import MergeChannelsTask, NoiseReduceTask, ExtractPeaksTask
from engine.features.feature_extraction import ExtractGeoFeaturesTask, ExtractHOGHWRecogTask, ExtractGeoHWRecogTask, \
    ExtractPixelsTask, RepeatFeaturesTask, ExtractHOGFeaturesTask
from .files.parallel import ParallelTask
from .helpers import read_file
from .hmm.recognition import RecognizeTask
from .hmm.reporting import ExtractionTask
from .hmm.training import CreateHMMModelTask, TrainModelTask
from .k_fold import KFoldSeparationTask, KFoldNNMergingTask
from .nn.recognition import RecognizeNNTask
from .nn.reporting import ExtractionJSONTask
from .nn.training import CreateNNModelTask
from .nn.training import TrainNNModelTask
from .noise import NoiseProfileLoadTask
from .processing.audio.extractlabel import ExtractLabelPartsTask
from .processing.audio.noise import NoiseProfileTask
from .processing.audio.silent import SilentAudioTask
from .processing.audio.splitseq import SplitIntoPartsTask
from .processing.filter import DurationLengthFilter, MinCountLabelFilter
from .spectrograms import CreateSpectrogramTask, BinarizeSpectrogramTask, SmartBinarizeSpectrogramTask
from .utils import print_log, write_log

TASKS = dict((t.__name__, t) for t in [
    # Audio tasks
    MergeChannelsTask, NoiseReduceTask,
    AudioLoadTask, AudioMatchingFilesLoadTask, AudioMatchTask,
    ExtractPeaksTask, SplitIntoPartsTask, ExtractLabelPartsTask, SilentAudioTask,
    DurationLengthFilter, MinCountLabelFilter,

    # Feature extraction tasks
    ExtractGeoFeaturesTask, ExtractHOGFeaturesTask, ExtractHOGHWRecogTask, ExtractGeoHWRecogTask, ExtractPixelsTask,

    # Noise profiling tasks
    NoiseProfileTask, NoiseProfileLoadTask,

    # Spectrogram tasks
    CreateSpectrogramTask, BinarizeSpectrogramTask, SmartBinarizeSpectrogramTask,

    # Model creation tasks
    CreateHMMModelTask, CreateNNModelTask,

    # Model training tasks
    TrainModelTask, TrainNNModelTask,

    # Recognition tasks
    RecognizeTask, RecognizeNNTask,
    ExtractionTask, ExtractionJSONTask,

    # K-Fold tasks
    KFoldSeparationTask, ParallelTask, KFoldNNMergingTask,

    # Other tasks
    RepeatFeaturesTask,
])


def import_pipeline(pipeline):
    """Import a pipeline from a structure"""
    if 'pipeline' in pipeline:
        pipeline = pipeline['pipeline']
    current_task = None
    for t in pipeline:
        props = t['props']
        if current_task is not None:
            props['src_list'] = current_task
        for k in props.keys():
            if isinstance(props[k], list) and len(props[k]) and isinstance(props[k][0], dict) and 'task' in props[k][0]:
                props[k] = import_pipeline(props[k])
        write_log(f"  {t['task']}({', '.join(f'{k}={repr(v)}' for k, v in props.items())})")
        current_task = TASKS[t['task']](**props).dest_list
    return current_task


def import_pipeline_file(file):
    """Import a pipeline from a json file"""
    print_log(f"Import pipeline from {file}")
    return import_pipeline(json.loads(read_file(file)))
