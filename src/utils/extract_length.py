import json
import os
from argparse import ArgumentParser
from pathlib import Path

import mutagen
from tqdm import tqdm

from engine.helpers import read_file, write_file


def run(in_file, out_file, data_path, ignore_missing=False):
    in_file = Path(in_file)
    data_path = Path(data_path)
    if out_file is None:
        out_file = in_file.with_suffix(f'.new{in_file.suffix}')
    else:
        out_file = Path(out_file)
    db = json.loads(read_file(in_file))

    for el in tqdm(db.values()):
        file_path = data_path / el['folder'] / el['filename']
        if ignore_missing and not file_path.is_file():
            continue
        if 'duration' not in el:
            f = mutagen.File(file_path)
            el['duration'] = f.info.length
        if 'size' not in el:
            el['size'] = os.path.getsize(file_path)

    write_file(out_file, json.dumps(db, indent='\t'))


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Extract duration and number of bytes from files\n\n'
                    'Gilles Waeber, VII 2019'
    )
    parser.add_argument('in_file', help='metadata file')
    parser.add_argument('out_file', nargs='?', default=None, help='metadata file')
    parser.add_argument('data_path', nargs='?', default='data', help='data folder')
    parser.add_argument('-i', '--ignore-missing', action='store_true', default='data', help='data folder')
    args = parser.parse_args()
    run(**vars(args))
