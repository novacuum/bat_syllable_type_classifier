import csv
import re
from argparse import ArgumentParser

from engine.helpers import TmpFile
from engine.reporting import add_to_csv


def run(dest_file, src_files):
    with TmpFile(dest_file) as out:
        for src in src_files:
            with open(src, 'r', newline='') as f:
                file_header = f.readline()
                assert re.match(r'^\W*"sep=,"\W*$', file_header), f"Invalid existing report file {file_header}"
                reader = csv.reader(f, dialect='excel')
                lines = list(reader)
                headers = lines[0]
                data = lines[1:]

                add_to_csv(out, headers, data)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Merge CSV files given in argument with the first argument (append if exists)\n\n'
                    'Gilles Waeber, VII 2019'
    )
    parser.add_argument('dest_file', help='destination file')
    parser.add_argument('src_files', nargs='+', help='source files')
    args = parser.parse_args()
    run(**vars(args))
