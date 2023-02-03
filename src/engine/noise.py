import sys
from os import path

from .files.files import File, FileType
from .files.lists import FileList
from .files.tasks import TransformationTask, SourceTask
from .settings import FILES_FOLDER
from .utils import list_files, mkdir, print_log, list_ellipsis, Progress, call_sox
from engine.helpers import TmpFile


# Noise reduction
def print_usage():
    print('Noise reduction')
    print('---------------')
    print('Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018\n')
    print('Generates noise profiles and applies noise reduction\n')
    print(f"Usage: See audio.py")
    sys.exit(-1)


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
            '-t', dest_file.suffix[1:],
            out,
            'noisered', profile, str(sensitivity)
        ], progress=progress)


# Create noise profiles from audio files
# not used!
class NoiseProfileTask(TransformationTask):
    def __init__(self, src_list, file_resolution):
        super().__init__(src_list, ProfileFileList([
            File(f"{f.folder}/no", f"{f.p.stem}.prof", f) for f in src_list.files
        ], self, file_resolution), {})

    def run(self, missing, progress=None):
        self.src_list.run()
        print_log('  Creating noise profiles')
        if progress is None:
            progress = Progress(len(missing))
        for file in missing:
            if path.exists(file.path()):
                progress.step(f"  {file.path()} already exists")
            else:
                mkdir(file.folder)
                generate_noise_profile(
                    file.task_src.path(), file.path(), progress=progress)


# Is the file an audio file
def is_profile(file):
    return path.splitext(file)[1] == '.prof'


# Load existing noise profiles
class NoiseProfileLoadTask(SourceTask):
    def __init__(self, folder, file_resolution):
        props = {'folder': folder, 'file_resolution': file_resolution}
        name = folder
        folder = f"{FILES_FOLDER}/{folder}"
        items = [f for f in list_files(folder) if is_profile(f)]
        print_log(f"Loaded noise profiles from {folder}")
        super().__init__(name, ProfileFileList(
            [File(folder, item) for item in items], self, file_resolution
        ), props)


class ProfileFileList(FileList):
    type = FileType.PROFILE

    def __init__(self, files, task, file_resolution):
        super().__init__(files, task)
        assert file_resolution == 'unique'  # One noise-reduction profile
        self.file_resolution = file_resolution
        assert len(files) == 1, f"File resolution mode is unique but there are {len(files)} files"

    def for_file(self, name):
        if self.file_resolution == 'unique':
            return self.files[0].path()
        else:
            raise ValueError(f"Unknown resolution mode {self.file_resolution}")


# Load existing noise profiles
def profiles(folder, file_resolution='label') -> ProfileFileList:
    task = NoiseProfileLoadTask(folder, file_resolution)
    return task.dest_list


if __name__ == "__main__":
    print_usage()
