import csv
import json
from argparse import ArgumentParser
from pathlib import Path

from engine.helpers import read_file
from engine.reporting import flatten_dict


def run(files):
    for file in files:
        file = Path(file)
        assert '.json' in file.suffixes, f"Expected '.json' in suffixes, got {file.suffixes}"
        data = json.loads(read_file(file))
        data = [flatten_dict(d) for d in data.values()]
        keys = list(k for d in data for k in d.keys())
        keys = list(dict.fromkeys(keys))  # Remove duplicates

        with open(file.with_name(f"{file.name.split('.')[0]}.csv"), 'w', newline='') as f:
            f.write('"sep=,"\r\n')
            writer = csv.writer(f, dialect='excel')
            writer.writerow(keys)
            writer.writerows([[row[k] if k in row else None for k in keys] for row in data])


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Convert a simple json file to csv\n\n'
                    'Gilles Waeber, VII 2019'
    )
    parser.add_argument('files', nargs='+', help='source files')
    args = parser.parse_args()
    run(**vars(args))
