from unittest import TestCase
from engine.processing.audio.silent import filter_and_transform_labels, filter_silent_parts
from engine.settings import BSC_ROOT_DATA_FOLDER
from utils.file import to_local_data_path, expand_data_path, to_unix_path


class Test(TestCase):
    def test_filter_silent_parts(self):
        silent_parts = filter_silent_parts([[2, 3], [5, 6], [8, 10]], filter_and_transform_labels([{'start': 1, 'end': 2}], 0, 3, 1))
        self.assertEqual(len(silent_parts), 2)
        self.assertEqual(silent_parts[0], [5, 6])
        self.assertEqual(silent_parts[1], [8, 10])

        silent_parts = filter_silent_parts([[2, 3], [5, 6], [8, 10]], filter_and_transform_labels([{'start': 4, 'end': 7}], 0, 5, 1))
        self.assertEqual(len(silent_parts), 2)
        self.assertEqual(silent_parts[0], [2, 3])
        self.assertEqual(silent_parts[1], [8, 10])

    def test_filter_and_transform_labels(self):
        labels = filter_and_transform_labels([{'start': 1, 'end': 2}], 0, 3, 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].start, 1)
        self.assertEqual(labels[0].end, 2)

        labels = filter_and_transform_labels([{'start': 1, 'end': 2}, {'start': 3, 'end': 4}], 2, 10, 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0].start, 3 - 2)
        self.assertEqual(labels[0].end, 4 - 2)

        param_labels = [{'start': 2, 'end': 3}, {'start': 3, 'end': 4}, {'start': 9, 'end': 11}]
        labels = filter_and_transform_labels(param_labels, 2, 10, 1)
        self.assertEqual(len(labels), 3)
        for i, label in enumerate(param_labels):
            self.assertEqual(labels[i].start, label['start'] - 2)
            self.assertEqual(labels[i].end, label['end'] - 2)

    def test_file(self):
        self.assertEqual(None, to_local_data_path(None))
        self.assertEqual('simple', str(to_local_data_path(BSC_ROOT_DATA_FOLDER/'simple')))
        self.assertEqual('simple', str(to_local_data_path(str(BSC_ROOT_DATA_FOLDER/'simple'))))
        self.assertEqual(BSC_ROOT_DATA_FOLDER / 'simple', expand_data_path(str(BSC_ROOT_DATA_FOLDER/'simple')))
        self.assertEqual('this/is/a/test', to_unix_path(r'this\is\a\test'))

