from engine.files.files import FileType, File
from engine.files.lists import FileList
from engine.files.tasks import PreprocessingTask
from engine.helpers import TmpFile
from engine.utils import call_sox


def generate_noise_profile(src_file, dest_file, progress=None):
    with TmpFile(dest_file) as out:
        call_sox([
            src_file,
            "-n", "noiseprof",
            out
        ], progress=progress)


def noise_reduce(sensitivity, src_file, dest_file, profile, progress=None):
    with TmpFile(dest_file) as out:
        call_sox([
            src_file,
            out,
            'noisered', profile, str(sensitivity)
        ], progress=progress)


# Create noise profiles from audio files
class NoiseProfileTask(PreprocessingTask):
    def __init__(self, src_list):
        super().__init__(src_list, ProfileFileList([
            File(f"{f.folder}/no", f"{f.p.stem}.prof", f) for f in src_list.files
        ], self), {})

    def run_file(self, file: File):
        file.p.parent.mkdir(parents=True, exist_ok=True)
        generate_noise_profile(file.task_src.path(), file.path())

    def extract_key(self, stem):
        return self.src_list.task.extract_key(stem)


class ProfileFileList(FileList):
    type = FileType.PROFILE

    def __init__(self, files, task):
        super().__init__(files, task)
        self.index = {file.p.stem: file for file in files}

    def for_file(self, file: File):
        return self.index[self.task.extract_key(file.p.stem)]
