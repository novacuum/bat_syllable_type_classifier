from engine.audio import AudioFileList, StartEndTime
from engine.files.files import File
from engine.files.tasks import PreprocessingTask
from engine.helpers import TmpFile
from engine.processing.audio.silent import SilentAudioList
from engine.utils import call_sox, call_piped_sox


def create_parent_path(file: File, fixed_length, dataset_name, padded_left):
    if dataset_name is None:
        return file.p.parent / ('label_' + ('fixed' if fixed_length else 'variable') + ('_left' if padded_left else ''))
    else:
        return file.p.parent.parent.parent / dataset_name / 'audio'


class ExtractLabelPartsTask(PreprocessingTask):
    def __init__(self, src_list: AudioFileList, fixed_length=False, padded_left=False, dataset_name=None):
        dest_files = []
        self.max_length = 0
        parent_path = create_parent_path(src_list.files[0], fixed_length, dataset_name, padded_left)

        for file in src_list.files:
            for label in file.metadata.labels:
                start = label['start']
                end = label['end']
                sequence = label['sequence']

                if fixed_length:
                    self.max_length = max(self.max_length, end - start)

                dest_files.append(File(
                    parent_path / f'{file.p.stem}_{start * 1000:09.2f}_{sequence}{file.p.suffix}',
                    task_src=StartEndTime(file, start, end),
                    metadata=file.metadata.as_slice_with_label(sequence, start, end, file.p.stem)
                ))
        super().__init__(src_list, AudioFileList(dest_files, self), {
            'fixed_length': fixed_length
            , 'dataset_name': dataset_name
            , 'padded_left': padded_left
        })
        self.silent_audio_list: SilentAudioList = src_list.create_silent_derivatives(
            self.max_length) if fixed_length else None

    def run(self, missing, *, parallel=None):
        if self.silent_audio_list is not None:
            self.silent_audio_list.run(parallel=parallel)
        super().run(missing, parallel=parallel)

    def run_file(self, file):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        src_file = file.task_src.file
        start_end_time: StartEndTime = file.task_src

        pad_after = max(0, self.max_length - (start_end_time.end - start_end_time.start))
        if pad_after == 0:
            extract_by_start_end_time(src_file.p, file.p, file.task_src, src_file.metadata.duration)
        else:
            extract_by_start_end_time_with_fixed_length(
                file.task_src.file.p, file.p, file.task_src, src_file.metadata.duration,
                self.max_length, self.silent_audio_list.for_file(src_file)
            )


def extract_by_start_end_time_with_fixed_length(src_file, dest_file, start_end_time: StartEndTime, duration, max_length, silent_audio_file):
    with TmpFile(dest_file) as out:
        call_piped_sox([
            '[bin:sox]',
            f'"|[bin:sox] {silent_audio_file} -p trim 0.001 {max_length - (start_end_time.end - start_end_time.start):.6f}"',
            f'"|[bin:sox] {src_file} -p trim {max(0.0, start_end_time.start):.6f} ={min(duration, start_end_time.end):.6f}"',
            '-t', dest_file.suffix[1:], out,
        ])

        # call_piped_sox([
        #     '[bin:sox]',
        #     src_file, '-p',
        #     'trim', f'{max(0.0, start_end_time.start):.6f}', f'={min(duration, start_end_time.end):.6f}',
        #     '|',
        #     '[bin:sox]',
        #     silent_audio_file,
        #     '-M', '-', '-p',
        #     '|',
        #     '[bin:sox]',
        #     '-',
        #     'trim', f'{-max_length:.6f}',
        #     '-t', dest_file.suffix[1:], out,
        # ])


def extract_by_start_end_time(src_file, dest_file, start_end_time: StartEndTime, duration):
    with TmpFile(dest_file) as out:
        call_sox([
            src_file,
            '-t', dest_file.suffix[1:], out,
            'trim', f'{max(0.0, start_end_time.start):.6f}', f'={min(duration, start_end_time.end):.6f}',
            # 'pad', f'{0:.0f}', f'{pad_after:.6f}',
        ])
