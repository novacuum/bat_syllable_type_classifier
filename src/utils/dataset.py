import datetime, math, random, csv
from pathlib import Path
from engine.metadata import metadata_db
from collections import defaultdict

from engine.utils import mkdir
from model.bat import BatService
from engine.settings import BSC_MS_DATETIME_FORMAT, BSC_ROOT_DATA_FOLDER


def to_audio_path(dataset, stem):
    return dataset + '/audio/' + stem + '.wav'


def create_dataset_config_simple(dataset, name, filter_callback, entry_filter=None, path_suffix=''):
    mdb = metadata_db(dataset + '/metadata.json')
    directory = BSC_ROOT_DATA_FOLDER / (dataset + '/config/' + name)
    label_count = defaultdict(lambda: 0)

    mkdir(directory)
    f_stem_list = []
    for stem in mdb.db:
        if entry_filter and not entry_filter(stem, mdb.db[stem]):
            continue
        f_stem_list.append(stem)

    for stem in f_stem_list:
        entry = mdb.db[stem]
        label_count[entry.label] += 1
    label_count = dict(filter(filter_callback, label_count.items()))
    min20 = list()

    for stem in f_stem_list:
        entry = mdb.db[stem]

        if entry.label not in label_count:
            continue

        min20.append(stem)

    with open(directory / 'all.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=['stem'])
        for stem in min20:
            writer.writerow({'stem': to_audio_path(dataset, path_suffix + stem)})

    with open(directory / 'test_meta.csv', 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile
            , delimiter=','
            , lineterminator='\n'
            , fieldnames=['label', 'count']
        )
        writer.writeheader()
        for label in label_count:
            writer.writerow({'label': label, 'count': label_count[label]})


def create_dataset_config(dataset, name, filter_callback):
    mdb = metadata_db(dataset + '/metadata.json')
    directory = BSC_ROOT_DATA_FOLDER / (dataset + '/config/' + name)
    label_count = defaultdict(lambda: 0)

    mkdir(directory)

    for stem in mdb.db:
        entry = mdb.db[stem]
        label_count[entry.label] += 1
    label_count = dict(filter(filter_callback, label_count.items()))

    batService = BatService()
    label_time_slice_order = dict()
    min20 = list()

    for stem in mdb.db:
        entry = mdb.db[stem]

        if entry.label not in label_count:
            continue

        min20.append(stem)
        if entry.label not in label_time_slice_order:
            label_time_slice_order[entry.label] = {'p1': list(), 'p2': list()}

        ms = datetime.datetime.strptime(entry.datetime, BSC_MS_DATETIME_FORMAT)
        bat = batService.get_by_id(entry.individual)
        if bat.get_half_time() > ms:
            label_time_slice_order[entry.label]['p1'].append(stem)
        else:
            label_time_slice_order[entry.label]['p2'].append(stem)

    test_elements = list()
    config_list = list()
    for sequence in label_time_slice_order:
        config = {'label': sequence, 'count': label_count[sequence], 'test_count': 0, 'validation_count': 0,
                  'validation_count_p1': 0, 'validation_count_p2': 0}
        config_list.append(config)

        for partition_label in ['p1', 'p2']:
            validation_number = math.ceil(len(label_time_slice_order[sequence][partition_label]) * .25)
            samples = random.sample(label_time_slice_order[sequence][partition_label], validation_number)
            config['validation_count_' + partition_label] = len(samples)
            config['validation_count'] += len(samples)
            test_elements += samples

        config['test_count'] = config['count'] - config['validation_count']

    with open(directory / 'test.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=['stem'])
        for stem in test_elements:
            writer.writerow({'stem': to_audio_path(dataset, stem)})

    with open(directory / 'train.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=['stem'])
        for stem in min20:
            if stem not in test_elements:
                writer.writerow({'stem': to_audio_path(dataset, stem)})

    with open(directory / 'all.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=['stem'])
        for stem in min20:
            writer.writerow({'stem': to_audio_path(dataset, stem)})

    with open(directory / 'test_meta.csv', 'w') as csvfile:
        writer = csv.DictWriter(
            csvfile
            , delimiter=','
            , lineterminator='\n'
            , fieldnames=['label', 'count', 'test_count', 'validation_count', 'validation_count_p1',
                          'validation_count_p2']
        )
        writer.writeheader()
        for config in config_list:
            writer.writerow(config)


def create_mp1_data_set(name, filter_callback):
    create_dataset_config('manual_peak_1', name, filter_callback)
