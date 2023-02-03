import datetime
import shutil, os
from pathlib import Path

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
SERVER_DIR = Path(__file__).resolve().parent.parent

# BirdVoice data folder
BIRDVOICE_BASE_DIR = SERVER_DIR.parent
BIRDVOICE_DATA_DIR = BIRDVOICE_BASE_DIR / 'data'

#  Globals
# -------
# Author: Gilles Waeber <moi@gilleswaeber.ch>, XI 2018
#
# Constants and helper functions used throughout the app


# Constants
HEREST_MIN_VARIANCE = 0.000001
TRAINING_ITERATIONS = 4
HTK_CONFIGURATION = "NATURALREADORDER=T\n" \
                    "NATURALWRITEORDER=T\n" \
                    "FORCEOUT=T\n"
# NATURALREADORDER, NATURALWRITEORDER: use PC byte ordering
# FORCEOUT: force HVite to produce an output even when it failed completely for a sample

AUDIO_EXTENSIONS = ['.wav', '.flac', '.mp3', '.ogg', '.aac']

# Data folder
# BIRDVOICE_FOLDER = os.getenv('BIRDVOICE_DATA', '.')
BIRDVOICE_FOLDER: Path = BIRDVOICE_DATA_DIR
# FILES_FOLDER: Path = BIRDVOICE_FOLDER / 'data'
FILES_FOLDER: Path = BIRDVOICE_FOLDER
UPLOAD_FOLDER: Path = FILES_FOLDER / 'upload'
# MODELS_FOLDER: Path = BIRDVOICE_FOLDER / 'models'
SOFTWARE_FOLDER: Path = SERVER_DIR / 'software'

# Set folder
BSC_ROOT_DATA_FOLDER = BIRDVOICE_DATA_DIR
BSC_DATA_FOLDER = BSC_ROOT_DATA_FOLDER / os.getenv('BSC_DATASET_NAME', 'simple_call_test')
BSC_CONFIG_FOLDER = BSC_DATA_FOLDER / 'config'
BSC_AUDIO_FOLDER = BSC_DATA_FOLDER / 'audio'
MODELS_FOLDER: Path = BSC_DATA_FOLDER / 'models'
# Results folder
RESULTS_FOLDER: Path = BIRDVOICE_FOLDER / 'results'

# Log file
LOG_FOLDER: Path = BIRDVOICE_FOLDER / 'log'
LOG_FILE: Path = LOG_FOLDER / f'log-{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt'

# datetime format
BSC_MS_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'


def change_data_source(name):
    global BSC_DATA_FOLDER, BSC_CONFIG_FOLDER, BSC_AUDIO_FOLDER, MODELS_FOLDER

    BSC_DATA_FOLDER = BSC_ROOT_DATA_FOLDER / name
    BSC_CONFIG_FOLDER = BSC_DATA_FOLDER / 'config'
    BSC_AUDIO_FOLDER = BSC_DATA_FOLDER / 'audio'
    MODELS_FOLDER = BSC_DATA_FOLDER / 'models'


def find_software_in_path(name) -> Path:
    location = shutil.which(name)
    if location is None:
        return find_local_software(name)
    return Path(location)


def find_local_software(name, directory=SOFTWARE_FOLDER) -> Path:
    candidates = list(directory.glob(name + ".exe"))
    if len(candidates) == 0:
        candidates = list(directory.glob(name + "*"))

    assert len(candidates) != 0, f"No candidate found for {name}"
    assert len(candidates) == 1, f"Multiple candidate found for {name}"
    if os.path.isdir(candidates[0]):
        return find_local_software(name, candidates[0])

    return candidates[0]


# Binaries and scripts
BIN_HHEd = find_software_in_path('HHEd')
BIN_HERest = find_software_in_path('HERest')
BIN_HVite = find_software_in_path('HVite')
BIN_HParse = find_software_in_path('HParse')
# BIN_HResults = find_software_in_path('HResults')  # Not used anymore
BIN_SOX = find_software_in_path('sox')
BIN_HWRECOG = find_local_software('HWRecog-*.jar')
