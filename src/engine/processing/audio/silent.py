import librosa, soundfile, numpy, math, re
from collections import defaultdict
from functools import reduce
from attr import dataclass
from engine.files.files import File, FileType
from engine.files.lists import FileList
from engine.files.tasks import PreprocessingTask
from engine.helpers import TmpFile
from engine.processing.audio.noise import NoiseProfileTask, ProfileFileList
from engine.spectrograms import SpectrogramFileList, CreateSpectrogramTask


def filter_and_transform_labels(labels, offset, duration, sr):
    start = offset
    end = offset + duration
    transformed_time_slices = []
    for label in labels:
        if start < label['end'] and end > label['start']:
            transformed_time_slices.append([label['start'] - offset, label['end'] - offset])

    return librosa.time_to_samples(sorted(transformed_time_slices, key=lambda slice: slice[0]), sr)


def filter_silent_parts(silent_parts, labels):
    filtered_silent_parts = []
    default_label = (-1, [99999998, 99999999])
    label_iter = enumerate(labels)
    label = next(label_iter, default_label)[1]

    for silent_part in silent_parts:
        while silent_part[0] > label[1]:
            label = next(label_iter, default_label)[1]

        if silent_part[1] < label[0]:
            filtered_silent_parts.append(silent_part)

    return filtered_silent_parts


def get_silent_parts(y, labels):
    """
    Adapted from @see librosa.effects.split
    """
    hop_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop_length)

    # Compute the MSE for the signal
    mse = rms ** 2
    mse = librosa.power_to_db(mse.squeeze(), ref=numpy.max, top_db=None)

    # Remove energy from foreground/labeled parts before calculating mean energy
    filtered_mse = []
    end = 0
    for label in librosa.samples_to_frames(labels, hop_length=hop_length):
        filtered_mse.extend(mse[end:label[0]])
        end = label[1]
    filtered_mse.extend(mse[end:])
    mse_threshold = numpy.mean(filtered_mse) + 1

    # Define all parts as silent which are lower than mean energy in DB + 1 DB
    silent = mse < mse_threshold

    # Interval slicing, adapted from
    # https://stackoverflow.com/questions/2619413/efficiently-finding-the-interval-with-non-zeros-in-scipy-numpy-in-python
    # Find points where the sign flips
    edges = numpy.flatnonzero(numpy.diff(silent.astype(int)))

    # Pad back the sample lost in the diff
    edges = [edges + 1]

    # If the first frame had low energy, count it
    if silent[0]:
        edges.insert(0, [0])

    # Likewise for the last frame
    if silent[-1]:
        edges.append([len(silent)])

    # Convert from frames to samples
    edges = librosa.frames_to_samples(numpy.concatenate(edges), hop_length=hop_length)

    # Clip to the signal duration
    edges = numpy.minimum(edges, y.shape[-1])

    # Stack the results back as an ndarray
    return edges.reshape((-1, 2)), mse_threshold


# Create silent audios from audio files
class SilentAudioTask(PreprocessingTask):
    def __init__(self, src_list, max_length):
        self.max_length = max_length
        self.key_regex = re.compile('(([^_]*_){2}(\d{2}_){3})(\d*)')
        index = defaultdict(list)

        for file in src_list.files:
            if len(file.metadata.labels) > 0:
                index[self.extract_key(file.p.stem)].append(file)

        super().__init__(src_list, SilentAudioList([
            File(f"{index[key][0].folder}/silent", f"{key}.wav", AudioFileSet(index[key])) for key in index
        ], self), {'max_length': max_length})

    def run_file(self, file: File):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        create_silent_audio(file.task_src, file.path(), self.max_length)

    def extract_key(self, stem):
        match = self.key_regex.findall(stem)
        return match[0][0] + 'combined'


class SilentAudioList(FileList):
    type = FileType.AUDIO
    task: SilentAudioTask

    def __init__(self, files, task: SilentAudioTask):
        super().__init__(files, task)
        self.index = {file.p.stem: file for file in files}

    def for_file(self, file: File):
        return self.index[self.task.extract_key(file.p.stem)]

    def noise_profile(self) -> ProfileFileList:
        task = NoiseProfileTask(self)
        return task.dest_list

    def create_spectrogram(self, **kwargs) -> SpectrogramFileList:
        task = CreateSpectrogramTask(self, **kwargs)
        return task.dest_list


class AudioFileSet:
    files: list

    def __init__(self, files):
        self.files = files
        self.len = reduce(lambda len, file: len + file.metadata.duration, files, 0)

    def load(self, offset, duration):
        y = numpy.empty(0, dtype=numpy.float)
        labels = list()
        end = start = sr = y_len = 0

        for file in self.files:
            end += file.metadata.duration

            if end > offset:
                labels.extend([{'start': label['start'] + start, 'end': label['end'] + start} for label in file.metadata.labels])
                y_temp, sr = librosa.load(file.path(), sr=None, offset=offset - start, duration=duration - y_len)
                y = numpy.concatenate((y, y_temp))
                y_len = len(y) / sr
                if y_len >= duration * sr:
                    break

            start = end

        return y, sr, filter_and_transform_labels(labels, offset, duration, sr)


@dataclass(frozen=True)
class TimeSlice:
    start: float
    end: float


def create_silent_audio(file_set: AudioFileSet, dest_file, max_length):
    silent_audio = None
    silent_audio_parts = numpy.empty(0, dtype=numpy.float)

    for i in range(1, 3):
        # Exit if file is shorter than offset
        if file_set.len < 10 * (i - 1):
            break

        time_slice = {'duration': 10 * i, 'offset': 10 * (i - 1)}

        y, sr, labels = file_set.load(**time_slice)
        pad = librosa.time_to_samples(.005, sr)
        silent_parts, threshold = get_silent_parts(y, labels)

        if len(silent_parts) > 0:
            silent_parts[...] += [pad, -pad]
            silent_parts = list(filter(lambda part: part[1] - part[0] > 10, silent_parts))

            if len(silent_parts) > 0:
                silent_parts = filter_silent_parts(silent_parts, labels)

            if len(silent_parts) > 0:
                silent_parts = sorted(silent_parts, key=lambda part: part[1] - part[0], reverse=True)

                # Check longest first
                silent_part = silent_parts[0]
                if silent_part[1] - silent_part[0] > max_length * sr:
                    silent_audio = y[silent_part[0]:silent_part[1]]
                    break

                # try to create a silent audio files from patches
                for silent_part in silent_parts:
                    silent_audio_parts = numpy.concatenate((silent_audio_parts, y[silent_part[0]:silent_part[1]]))
                    if len(silent_audio_parts) > max_length * sr:
                        silent_audio = silent_audio_parts
                        break

    if silent_audio is None:
        if len(silent_audio_parts) > 10:
            silent_audio = silent_audio_parts
            while len(silent_audio) < max_length * sr:
                silent_audio = numpy.concatenate((silent_audio, silent_audio_parts))
        else:
            # print(f'no silent parts for {src_file}')
            # little hacky, but sr should exists
            silent_audio = numpy.zeros(math.ceil(max_length * 2 * sr))

    # print(f'length {len(silent_audio)/sr:06.6f} from needed {max_length:1.6f}')

    with TmpFile(dest_file) as out:
        soundfile.write(out, silent_audio, sr, soundfile.default_subtype('WAV'), None, 'WAV')
