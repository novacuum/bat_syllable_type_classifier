import json
from argparse import ArgumentParser
from pathlib import Path

from engine.helpers import read_file, write_file


def run(dest_file, src_files):
    dest_file = Path(dest_file)
    if dest_file.is_file():
        data = json.loads(read_file(dest_file))
        assert 'results' in data
    else:
        data = {'results': {}}
    for f in src_files:
        n = json.loads(read_file(f))
        data['results'].update(n['results'])
    write_file(dest_file, json.dumps(data, indent='\t'))


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Merge JSON result files given in argument with the first argument (append if exists)\n\n'
                    'Gilles Waeber, VII 2019'
    )
    parser.add_argument('dest_file', help='destination file')
    parser.add_argument('src_files', nargs='+', help='source files')
    args = parser.parse_args()
    run(**vars(args))
