from __future__ import annotations

from dataclasses import dataclass
from os import path
from typing import Union, Sequence

from tqdm import tqdm

from .configuration import conf, to_conf_file_path
from .files.files import File, FileType
from .files.lists import FileList
from .files.parallel import ParallelFileList
from .files.tasks import PreprocessingTask, TransformationTask, SourceTask
from .files.tasks import VirtualTransformationTask
from .helpers import TmpFile
from .metadata import MetadataDB, metadata_db
from .noise import noise_reduce
from .processing.audio.silent import SilentAudioTask, SilentAudioList
from .settings import FILES_FOLDER, AUDIO_EXTENSIONS
from .spectrograms import CreateSpectrogramTask, SpectrogramFileList, FeaturesFileList
from .utils import list_files, mkdir, print_log, list_ellipsis, call_sox, write_log

f"""Audio import

Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018
Import audio files, merge channels

Usage:
  from {__package__}.audio import audio, audio_matching

  a = ...  # An audio collection
  am = ... # An audio collection that will reproduce the preprocessing steps of another audio chain
  c = ...  # A configuration filter (see configuration.py)
  s = ...  # A spectrogram collection (see spectrograms.py)
  p = ...  # A noise profile list
  folder = 'audio' # A reference to a directory inside the data folder
  sensitivity = 21 # Noise reduction sensitivity, in %

  a = load_audio(folder) # Create a track list
  a = load_audio(folder, c) # Create a track list, filtered using a configuration

  a = a.merge_channels() # Merge the audio channels
  p = a.noise_profile() # Turn audio into a noise profile list
  s = a.create_spectrogram() # Create audio spectrograms
  a = a.noise_reduce(p, sensitivity) # Reduce noise using a profile, with a given sensitivity

Shortcuts:
  a.noise_reduce(am, sensitivity) # Reduce noise, automatic profile creation
  a.geo_features(threshold) # Same as a.create_spectrogram().binarize(threshold).geo_features()
"""


def merge_channels(src_file, dest_file):
    with TmpFile(dest_file) as out:
        call_sox([
            src_file,
            '-c', '1',  # Convert to single channel
            out
        ])


def extract_peak(src_file, dest_file, peak, pad_before, pad_after, duration):
    with TmpFile(dest_file) as out:
        call_sox([
            src_file,
            '-t', dest_file.suffix[1:], out,
            'trim', f'{max(0, peak - pad_before):.3f}', f'={min(duration, peak + pad_after):.3f}',
            'pad', f'{max(0, pad_before - peak):.3f}', f'{max(0, peak + pad_after - duration):.3f}',
        ])


class MergeChannelsTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, AudioFileList([
            File(f"{f.folder}/mc", f.name, f) for f in src_list.files
        ], self), {})

    def __str__(self):
        return 'Merging channels'

    def run_file(self, file: File):
        mkdir(file.folder)
        merge_channels(file.task_src.path(), file.path())


class NoiseReduceTask(PreprocessingTask):
    def __init__(self, src_list, noise, sensitivity):
        if isinstance(noise, AudioFileList):
            noise = noise.preproc_from(src_list).noise_profile()
        self.noise = noise
        super().__init__(src_list, AudioFileList([
            File(f"{f.folder}/nr{sensitivity}", f.name, f) for f in src_list.files
        ], self), {'sensitivity': sensitivity, 'noise': noise.task.export()})

    def reuse(self, src_list):
        return NoiseReduceTask(src_list, self.noise, self.props['sensitivity'])

    def run(self, missing, *, parallel=None):
        self.src_list.run(parallel=parallel)
        print_log(f"  Reducing noise with {self.props['sensitivity']}% sensitivity")
        self.noise.run(parallel=parallel)
        for file in tqdm(missing):
            if path.exists(file.path()):
                write_log(f"  {file.path()} already exists")
            else:
                mkdir(file.folder)
                noise_reduce(self.props['sensitivity'] / 100, file.task_src.path(), file.p,
                             self.noise.for_file(file))


@dataclass(frozen=True)
class Peak:
    file: File
    peak: float


@dataclass(frozen=True)
class StartEndTime:
    file: File
    start: float
    end: float

    def to_metadata_args(self):
        return {
            'start': self.start
            , 'end': self.end
            , 'source_file_stem': self.file.p.stem
        }


class ExtractPeaksTask(TransformationTask):
    def __init__(self, src_list, pad_before, pad_after):
        self.pad_before = pad_before
        self.pad_after = pad_after
        dest_files = []
        ba_suffix = f'_b{pad_before * 1000:.0f}_a{pad_after * 1000:.0f}'

        for file in src_list.files:
            for peak in file.metadata.peaks:
                dest_files.append(File(
                    file.p.parent / 'peaks' / f'{file.p.stem}{ba_suffix}{file.p.suffix}',
                    task_src=Peak(file, peak),
                    metadata=file.metadata
                ))
        super().__init__(src_list, AudioFileList(dest_files, self), {
            'pad_before': pad_before,
            'pad_after': pad_after,
        })

    def run_file(self, file):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        extract_peak(file.task_src.file.p, file.p, file.task_src.peak, self.pad_before, self.pad_after,
                     file.metadata.duration)


class SplitAudioIntoPeaksTask(TransformationTask):
    def __init__(self, src_list, dataset_name, pad_before, pad_after):
        self.pad_before = pad_before
        self.pad_after = pad_after
        dest_files = []

        for file in src_list.files:
            for peak in file.metadata.peaks:
                timestamp = peak['timestamp']
                sequence = peak['sequence']

                dest_files.append(File(
                    file.p.parent.parent / dataset_name / 'audio' / f'{file.p.stem}_peak{timestamp * 1000:.0f}_{sequence}{file.p.suffix}',
                    task_src=Peak(file, timestamp),
                    metadata=file.metadata.with_syllable_label(peak['sequence'])
                ))
        super().__init__(src_list, AudioFileList(dest_files, self), {
            'pad_before': pad_before,
            'pad_after': pad_after,
        })

    def run_file(self, file):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        extract_peak(file.task_src.file.p, file.p, file.task_src.peak, self.pad_before, self.pad_after,
                     file.metadata.duration)


class AudioLoadTask(SourceTask):
    def __init__(self, mdb: MetadataDB, dataset_name, source):
        mdb = metadata_db(mdb)

        props = {'mdb': mdb.props(), 'dataset_name': dataset_name, 'source': source}
        name = source
        if not isinstance(source, str):  # List of files
            tracks = [FILES_FOLDER / f for f in source]
        elif path.isdir(FILES_FOLDER / source):
            # because model names contains the digest of the feature file list, sort audio files
            tracks = [FILES_FOLDER / f"{source}/{f}" for f in sorted(list_files(f"{FILES_FOLDER}/{source}")) if is_audio(f)]
        elif path.isfile(to_conf_file_path(dataset_name, source)):
            cnf = conf(dataset_name, source)
            name = cnf.name
            tracks = cnf.load()
            props['source'] = cnf.props()
        else:
            raise ValueError(f'Source path does not exist: {source}')
        print_log(f"Loaded audio tracks from {source}")

        super().__init__(name, AudioFileList(
            [File(track, metadata=mdb.for_file(track)) for track in tracks], self
        ), props)


class AudioMatchingFilesLoadTask(SourceTask):
    def __init__(self, files):
        props = {'files': files}
        print_log(
            f"Loaded audio (matching) tracks from files: {list_ellipsis(files)}")
        super().__init__('files', AudioFileList(
            [File(f) for f in files], self
        ), props)


class AudioMatchTask(VirtualTransformationTask):
    def __init__(self, src_list, copy_list):
        tasks = []
        previous = copy_list

        # Get previous tasks
        while isinstance(previous.task, TransformationTask):
            if isinstance(previous.task, PreprocessingTask):
                tasks.append(previous.task)
            previous = previous.task.src_list

        new_list = src_list
        for t in reversed(tasks):
            task = t.reuse(new_list)
            new_list = task.dest_list

        super().__init__(src_list, new_list, {'copy_list': copy_list.task.export()})


# Is the file an audio file
def is_audio(file):
    return path.splitext(file)[1] in AUDIO_EXTENSIONS


class AudioFileList(FileList):
    type = FileType.AUDIO

    def __init__(self, files, task):
        super().__init__(files, task)

    def preproc_from(self, from_list) -> FeaturesFileList:
        """Import the preprocessing pipeline"""
        task = AudioMatchTask(self, from_list)
        return task.dest_list

    def multi(self) -> AudioFileList:
        """Repeat the under-represented classes"""
        from engine.features.feature_extraction import RepeatFeaturesTask
        task = RepeatFeaturesTask(self)
        return task.dest_list

    def k_fold(self, k, val_bins=1, test_bins=1) -> Union[ParallelFileList, FeaturesFileList, AudioFileList]:
        """Use transparent K-Fold

        After a call to this method, all other actions are realized in parallel over all folds.
        The validation and the testing dataset are automatically injected.
        See the KFoldParallelList wrapper for more information."""
        from .k_fold import KFoldSeparationTask
        task = KFoldSeparationTask(self, k, val_bins=val_bins, test_bins=test_bins)
        return task.dest_list

    def merge_channels(self) -> AudioFileList:
        task = MergeChannelsTask(self)
        return task.dest_list

    def duration_length_filter(self, min_duration_length) -> AudioFileList:
        from .processing.filter import DurationLengthFilter
        task = DurationLengthFilter(self, min_duration_length)
        return task.dest_list

    def min_count_label_filter(self, min_count) -> AudioFileList:
        from .processing.filter import MinCountLabelFilter
        task = MinCountLabelFilter(self, min_count)
        return task.dest_list

    def noise_reduce(self, noise, sensitivity) -> AudioFileList:
        task = NoiseReduceTask(self, noise, sensitivity)
        return task.dest_list

    def extract_peaks(self, pad_before, pad_after) -> AudioFileList:
        task = ExtractPeaksTask(self, pad_before, pad_after)
        return task.dest_list

    def split_audio_into_peaks(self, dataset_name, pad_before, pad_after):
        task = SplitAudioIntoPeaksTask(self, dataset_name, pad_before, pad_after)
        return task.dest_list

    def extract_label_parts(self, fixed_length=False, padded_left=False, dataset_name=None) -> AudioFileList:
        from .processing.audio.extractlabel import ExtractLabelPartsTask
        task = ExtractLabelPartsTask(self, fixed_length, padded_left, dataset_name)
        return task.dest_list

    def split_into_parts(self, part_length, strides, label_min_cover_length) -> AudioFileList:
        from .processing.audio.splitseq import SplitIntoPartsTask
        task = SplitIntoPartsTask(self, part_length, strides, label_min_cover_length)
        return task.dest_list

    def create_silent_derivatives(self, max_length) -> SilentAudioList:
        task = SilentAudioTask(self, max_length)
        return task.dest_list

    def create_spectrogram(self, **kwargs) -> SpectrogramFileList:
        """
        :param height: for optimal performance sox needs a height which is power of 2 + 1
        :param kwargs:
        """
        task = CreateSpectrogramTask(self, **kwargs)
        return task.dest_list

    def geo_features(self, threshold) -> FeaturesFileList:
        return self.create_spectrogram().binarize(threshold).geo_features()

    def geo_hwr_features(self, threshold) -> FeaturesFileList:
        """Geometric features"""
        return self.create_spectrogram(height=256).binarize(threshold).geo_hwr_features()

    def hog_hwr_features(self) -> FeaturesFileList:
        """Histogram of Oriented Gradients"""
        return self.create_spectrogram(height=256).hog_hwr_features()


def load_audio(mdb: MetadataDB, dataset_name, source: Union[str, Sequence[str]]) -> AudioFileList:
    """Get audio files from a folder or a configuration file"""
    task = AudioLoadTask(mdb, dataset_name, source)
    return task.dest_list


def load_audio_matching_files(files) -> AudioFileList:
    """Get audio files, add the matching attribute"""
    task = AudioMatchingFilesLoadTask(files)
    return task.dest_list
