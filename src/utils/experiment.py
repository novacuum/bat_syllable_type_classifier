import argparse, nbformat
from pathlib import Path

from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from engine.audio import load_audio
from engine.k_fold import KFoldSeparationTask
from engine.metadata import metadata_db
from engine.settings import BSC_DATA_FOLDER, BIRDVOICE_BASE_DIR, BSC_ROOT_DATA_FOLDER
from engine.spectrograms import SpectrogramFileList
from scripts.models.cnn1d_model import test_cnn_1d_model
from scripts.models.cnn_model import test_cnn_model
from scripts.models.config import learning_rate, save_every, hog_num_bins
from scripts.models.datasets import Dataset
from scripts.models.densnet_model import test_densnet_model
from scripts.models.lstm_model import test_lstm_model
from utils.function import get_functions


def call_variant_function(module_name, ppl):
    descr = """experiment runner"""
    parser = argparse.ArgumentParser(description=descr, epilog='\n')
    parser.add_argument('-v', '--variant', help='variant name of the experiment')
    parser.add_argument('-i', '--index', help='config_index of the selected experiment variant', default=0)
    args = parser.parse_args()

    run_config_db = {name: function for name, function in get_functions(module_name, 'run_variant_')}
    run_config_db[f'run_variant_{args.variant}'](ppl, int(args.index))


def create_dataset_by_kfold_task(k_fold_task: KFoldSeparationTask, name, config):
    labels = list(k_fold_task.by_label.keys())
    print('create Dataset with:', labels)
    return Dataset(name, labels, config)


def run_basic_models(var_data, dataset, prefix, max_epochs, ignore_densnet=False):
    for f, t in var_data.items():
        common_params = dict(
            dataset=dataset,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            training=t,
            # validation=None,  K-Fold
            # testing=None,  K-Fold
            save_every=save_every,
            prepare_args=dict(),
            model_suffix=f'{prefix}{f}_{max_epochs}'
        )

        lstm_params = dict(
            **common_params,
            lstm_neurons=256,
        )

        test_lstm_model(**lstm_params)
        test_cnn_model(**common_params)
        test_cnn_1d_model(**common_params)
        if not ignore_densnet:
            test_densnet_model(**common_params)

        if 'hog' in f:
            params_3d = dict(**common_params)
            params_3d['prepare_args']['new_dim_size'] = hog_num_bins
            params_3d['model_suffix'] += '_3d'
            test_cnn_model(**params_3d)


def create_basic_features_and_kfold(ppl: SpectrogramFileList):
    var_data = {
        'raw': ppl.img_features(),
        'hog': ppl.hog_features()
    }

    for f, t in var_data.items():
        var_data[f] = t.k_fold(k=8, val_bins=1, test_bins=2)

    return var_data


def create_result_report_notebook_by_dataset(dataset: Dataset):
    create_result_report_notebook('testing_' + dataset.name)


def create_result_report_notebook(result_name):
    ep = ExecutePreprocessor()
    notebook_template_path = str(BIRDVOICE_BASE_DIR / 'experiments' / 'result_template.ipynb')
    out_path_html = str(BSC_ROOT_DATA_FOLDER / 'results' / 'report' / (result_name + '.html'))
    nb = nbformat.read(notebook_template_path, as_version=4)

    Path(notebook_template_path).parent.mkdir(parents=True, exist_ok=True)

    mapping = {
        'result_name': result_name
    }
    code_cell = "\n".join("{} = {}".format(key, repr(value)) for key, value in mapping.items())
    nb['cells'].insert(1, nbformat.v4.new_code_cell(code_cell))
    nb["metadata"].update({"hide_input": True})

    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': str(BIRDVOICE_BASE_DIR / 'experiments')}})
    except:
        # Execution failed, print a message then raise.
        msg = ('Error executing the notebook "%s".\n'
               'Notebook arguments: %s\n\n'
               'See notebook "%s" for the traceback.' %
               (notebook_template_path, str(mapping), out_path_html))
        print(msg)
        raise
    finally:
        for cell in nb.cells:
            cell.transient = {'remove_source': True}

        html_exporter = HTMLExporter()
        body, resources = html_exporter.from_notebook_node(nb)
        with open(str(out_path_html), 'wt', encoding='utf-8') as f:
            f.write(body)


def load_audio_and_db(dataset_name, source='audio'):
    assert str(BSC_DATA_FOLDER).endswith(dataset_name), 'invalid data folder'
    return load_audio(metadata_db(dataset_name + '/metadata.json'), dataset_name, f'{dataset_name}/{source}')
