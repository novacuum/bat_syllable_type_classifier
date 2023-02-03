import librosa
import numpy
from engine.audio import AudioFileList, StartEndTime
from engine.files.files import File
from engine.files.tasks import PreprocessingTask
from engine.helpers import TmpFile
from engine.metadata import Metadata
from engine.utils import call_sox
from utils.file import file_has_labels

SECONDS_TO_MICRO = 1000000
MICROSECONDS_TO_SECONDS = .000001
MICROSECONDS_TO_MILLI = 0.001


class SplitIntoPartsTask(PreprocessingTask):
    def __init__(self, src_list, part_length, strides, label_min_cover_length=0.8):
        int_strides = strides * SECONDS_TO_MICRO
        int_part_length = part_length * SECONDS_TO_MICRO
        directory_name = create_identifier(part_length, strides, label_min_cover_length)

        assert float(int(int_strides)) == int_strides, f"strides value ({int_strides:09.3f}µs): nanoseconds precision is not supported"
        assert float(int(int_part_length)) == int_part_length, f"part_length value ({int_part_length:09.3f}µs): nanoseconds precision is not supported"

        dest_files = self.with_label(
            src_list, int_part_length, int_strides, label_min_cover_length, directory_name
        ) if file_has_labels(src_list.files[0]) else self.without_label(
            src_list, part_length, strides, directory_name
        )

        super().__init__(src_list, AudioFileList(dest_files, self), {
            'part_length': part_length
            , 'strides': strides
            , 'label_min_cover_length': label_min_cover_length
        })

    @staticmethod
    def without_label(src_list, part_length, strides, directory_name):
        dest_files = []
        for file in src_list.files:
            for start in numpy.arange(0, librosa.get_duration(filename=file.path()) - part_length, strides):
                start_end_time = StartEndTime(file, start, start + part_length)
                dest_files.append(File(
                    file.p.parent / directory_name / f'{file.p.stem}_{start * 1000:06.0f}_test{file.p.suffix}',
                    task_src=start_end_time,
                    metadata=Metadata('none', **start_end_time.to_metadata_args())
                ))

        return dest_files

    @staticmethod
    def with_label(src_list, int_part_length, int_strides, label_min_cover_length, directory_name):
        dest_files = []

        for file in src_list.files:
            labels = file.metadata.labels
            labels.sort(key=lambda label: label['start'])

            current_label = LabelDto(enumerate(labels), int_part_length, label_min_cover_length)
            current_part = Part(int_part_length)
            none_counter = 0
            int_file_duration = file.metadata.duration * SECONDS_TO_MICRO

            while current_part.end < int_file_duration:
                empty_label = False

                # label before start
                if current_label.start < current_part.start:
                    # label is less than 80% inside the part
                    if current_label.min_duration > current_label.end - current_part.start:
                        current_label = current_label.next()
                        continue
                else:
                    if current_label.start < current_part.end:
                        # label is less than 80% inside the part
                        if current_label.min_duration > current_part.end - current_label.start:
                            empty_label = True
                    else:
                        empty_label = True

                label = 'none' if empty_label else current_label.label

                if label == 'none':
                    none_counter += 1
                    if none_counter > 3 and current_part.end + (2 * int_part_length) < current_label.start:
                        current_part.start += int_strides
                        continue
                else:
                    none_counter = 0

                start_end_time = StartEndTime(file, current_part.start * MICROSECONDS_TO_SECONDS, current_part.end * MICROSECONDS_TO_SECONDS)
                dest_files.append(File(
                    file.p.parent / directory_name / f'{file.p.stem}_{current_part.start * MICROSECONDS_TO_MILLI:06.0f}_{label}{file.p.suffix}',
                    task_src=start_end_time,
                    metadata=file.metadata.as_slice_with_label(label, **start_end_time.to_metadata_args())
                ))

                current_part.start += int_strides

        return dest_files

    def run_file(self, file):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        extract_by_start_end_time(file.task_src.file.p, file.p, file.task_src)


def create_identifier(part_length, strides, label_min_cover_length):
    return f'parts_pl{part_length * 1000:.1f}ms_s{strides * 1000:.1f}ms' + _create_suffix('_cl', label_min_cover_length)


def create_short_identifier(part_length, strides, label_min_cover_length):
    return f'p{part_length * 1000:.0f}s{strides * 1000:.0f}' + _create_suffix('l', label_min_cover_length)


def _create_suffix(prefix, label_min_cover_length):
    return '' if label_min_cover_length is None else f'{prefix}{label_min_cover_length * 100:.0f}%'


def extract_by_start_end_time(src_file, dest_file, start_end_time):
    with TmpFile(dest_file) as out:
        call_sox([
            src_file,
            '-t', dest_file.suffix[1:], out,
            'trim', f'{start_end_time.start:.6f}', f'={start_end_time.end:.6f}',
        ])


class LabelDto:
    __slots__ = ('label_iter', 'label', 'start', 'end', 'duration', 'min_duration', 'part_length', 'min_cover_length')

    def __init__(self, label_iter: enumerate, part_length, min_cover_length):
        label = next(label_iter, (-1, {'sequence': 'none', 'start': 999980, 'end': 999989}))[1]

        self.label_iter = label_iter
        self.label = label['sequence']
        self.start = label['start'] * SECONDS_TO_MICRO
        self.end = label['end'] * SECONDS_TO_MICRO

        self.duration = self.end - self.start
        self.min_duration = min(self.duration, part_length) * min_cover_length

        self.part_length = part_length
        self.min_cover_length = min_cover_length

    def next(self):
        return LabelDto(self.label_iter, self.part_length, self.min_cover_length)


class Part:
    def __init__(self, part_length):
        self.start = 0
        self.part_length = part_length

    @property
    def end(self):
        return self.start + self.part_length
