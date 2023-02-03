import itertools, numpy

from model.pipeline import JsonPipeline
from plot.operator import ModelMatcher, Mapper
from plot.settings import MODEL_NAMES_LONG, MODELS, MODEL_NAMES
from utils.format import latex_statistical_number
from utils.report import get_best_model_pipeline_from_result
from utils.task import create_shortened_identifier

result_table_fields = list(map("".join, itertools.product(*[['val', 'test'], ['_acc', '_loss'], ['_m', '_d']])))
table_field_name_mapping = {
    'val_acc': 'Validation accuracy'
    , 'val_loss': 'Validation loss'
    , 'test_acc': 'Test accuracy'
    , 'test_loss': 'Test loss'
}


class TestingResult:
    id: str
    title: str
    dataset: str
    features: str
    code: str
    results: str
    parameters: str
    result: dict

    def __init__(self, result):
        result['id'] = result['id'].replace('lstm2', 'lstm')
        self.result = result
        self.pipeline = JsonPipeline(result['pipeline'])
        self.id = result['id']
        self.extract_results()
        self.features = 'HOG' if self.pipeline.has_task('ExtractHOGHWRecogTask') else 'raw'
        self.key = self.id.split('/')[0].replace('%', '')

        if '_3d' == self.key[-3:]:
            self.features += ' 3D'

        model_name_index = MODELS.index(ModelMatcher(Mapper()).get(result))
        model_name = MODEL_NAMES_LONG[model_name_index]
        # currently all tests include K-Fold cross-validation
        self.title_short = MODEL_NAMES[model_name_index]
        self.title = model_name + ', using k-fold cross-validation'

    def get_stat_table_dict(self):
        result = dict()
        for field in result_table_fields:
            result[field] = self.result[field]
        return result

    def get_stat_table_row(self, *identifier_include):
        row = [self.get_stat_table_identifier(identifier_include)]
        row.extend([self.result[field] for field in result_table_fields])
        return row

    def get_stat_table_identifier(self, include):
        description = []

        if 'nr_sensitivity' in include:
            sensitivity = self.pipeline.get_task('NoiseReduceTask')['props']['sensitivity'] if self.pipeline.has_task('NoiseReduceTask') else 0
            description.append(f'\gls{{nrs}}: {sensitivity}')

        if 'split' in include:
            props = self.pipeline.get_task('SplitIntoPartsTask')['props']
            description.append(f'stride: {props["strides"]}, \gls{{mcl}}: {props["label_min_cover_length"] if "label_min_cover_length" in props else props["part_length"]}')

        if 'spectrogram' in include and self.pipeline.has_task('CreateSpectrogramTask'):
            props = self.pipeline.get_task('CreateSpectrogramTask')['props']
            xpps = str(props["x_pixels_per_sec"]).replace('000', 'K')
            description.append(f'\gls{{xpps}}: {xpps}, height: {props["height"]}')

        if 'features' in include:
            description.append(self.features.replace('HOG', '\gls{hog}'))

        return f'\\cite{{{self.key}}} {self.title_short}, {", ".join(description)}'

    @property
    def dataset(self):
        return ' '.join(self.pipeline.get_dataset_name().split('_')).title()

    def to_bib(self, num):
        return (f'@model{{{self.key},\n'
                + f'\tid = {{{self.id}}},\n'
                + f'\ttitle = {{{self.title}}},\n'
                + f'\tdataset = {{{self.dataset}}},\n'
                + f'\tfeatures = {{{self.features}}},\n'
                + (f'\tresults = {{{self.results}}},\n' if self.results is not None else '')
                + (f'\tparameters = {{{self.parameters}}},\n' if self.parameters is not None else '')
                + f'\tnum = {{{num}}}\n}}')

    def extract_results(self):
        result = self.result

        if 'val_acc_m' in result:
            self.results = f'validation: {latex_statistical_number(result["val_acc_m"], result["val_acc_d"], True)}'
            if 'test_acc_m' in result:
                self.results += f', testing: {latex_statistical_number(result["test_acc_m"], result["test_acc_d"], True)}'
            self.results = self.results.replace('%', '\\%')
        if 'epoch_m' in result:
            self.parameters = f'epochs: {latex_statistical_number(result["epoch_m"], result["epoch_d"], False)} '\
                              f'({", ".join(f"${e}$" for e in sorted(result["epoch"]))})'

