from collections import defaultdict
from engine.audio import AudioFileList
from engine.files.tasks import VirtualTransformationTask


class DurationLengthFilter(VirtualTransformationTask):
    def __init__(self, src_list: AudioFileList, min_duration_length):
        if 'labels' in src_list.files[0].metadata:
            dest_files = []
            for file in src_list.files:
                props = file.metadata.props
                props['labels'] = list(
                    filter(lambda label: label['end'] - label['start'] >= min_duration_length, props['labels']))
                if len(props['labels']) > 0:
                    dest_files.append(file)
        else:
            dest_files = list(filter(lambda file: file.metadata.duration > min_duration_length, src_list.files))

        super().__init__(src_list, AudioFileList(dest_files, self), {
            'min_duration_length': min_duration_length
        })


class MinCountLabelFilter(VirtualTransformationTask):
    def __init__(self, src_list: AudioFileList, min_count=50):
        if 'labels' in src_list.files[0].metadata:
            label_count = defaultdict(lambda: 0)
            dest_files = []

            for file in src_list.files:
                for label in file.metadata.labels:
                    label_count[label['sequence']] += 1

            label_count_filtered = dict(filter(lambda elem: elem[1] > min_count, label_count.items()))

            for file in src_list.files:
                props = file.metadata.props
                props['labels'] = list(filter(lambda label: label['sequence'] in label_count_filtered, props['labels']))

                if len(props['labels']) > 0:
                    dest_files.append(file)
        else:
            label_count = defaultdict(lambda: 0)
            for file in src_list.files:
                label_count[file.metadata.label] += 1

            label_count_filtered = dict(filter(lambda elem: elem[1] > min_count, label_count.items()))
            dest_files = list(filter(lambda file: file.metadata.label in label_count_filtered, src_list.files))

        print('label count', dict(label_count))
        print('label count filtered:', label_count_filtered)
        super().__init__(src_list, AudioFileList(dest_files, self), {
            'min_count': min_count
        })
