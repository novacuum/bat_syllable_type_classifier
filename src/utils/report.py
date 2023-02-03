import itertools, json, math, operator, os, random, base64
import numpy as np
import pandas as pd
from pathlib import Path

from IPython.display import HTML, display, Image, Markdown
from tensorflow import RaggedTensor

from engine.helpers import read_file
from engine.pipelines import import_pipeline_file
from engine.settings import BSC_ROOT_DATA_FOLDER, MODELS_FOLDER, RESULTS_FOLDER, BSC_DATA_FOLDER
from model.report import ReportResult
from plot.heatmap import heatmap, annotate_heatmap
from utils.file import expand_data_path
from utils.template import parse_template


def split_acc_loss_plots(plt, epochs, acc, val_acc, loss, val_loss, figsize=(8, 2)):
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    ax = axes[0]
    # ax.set_yscale('log')
    ax.plot(epochs, acc, label='training')
    ax.plot(epochs, val_acc, label='validation')
    ax.set_ylabel('accuracy'), ax.set_xlabel('epochs')
    ax.legend(), ax.grid()
    ax = axes[1]
    # ax.set_yscale('log')
    ax.plot(epochs, loss, label='training')
    ax.plot(epochs, val_loss, label='validation')
    ax.set_ylabel('loss (categorical cross-entropy)'), ax.set_xlabel('epochs')
    ax.legend(), ax.grid()
    fig.tight_layout()
    return fig, axes


def confusion_matrix(plt, matrix: dict):
    classes = matrix.keys()
    fig, ax = plt.subplots(figsize=(2.5, 2))
    data = np.array([[matrix[tc][pc] for pc in classes] for tc in classes])
    im, cbar = heatmap(data, classes, classes, cmap='viridis', ax=ax)  # afmhot_r
    cbar.locator.set_params(min_n_ticks=6)  # I wish more numbers...
    annotate_heatmap(im, data, valfmt='{x:.0f}', textcolors=["white", "black"])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    return fig


def nn_cnn_2d_lrp(epoch_file_path, selection=None):
    from engine.pipelines import import_pipeline_file
    ppl = import_pipeline_file(epoch_file_path)
    return nn_cnn_2d_lrp_by_pipeline(ppl, selection)


def nn_cnn_2d_lrp_by_pipeline(ppl, selection=None):
    from engine.k_fold import get_kfold_set
    from ext.heatmaps2 import LRP
    from engine.nn.training import find_best_models
    from tensorflow.keras.models import load_model, Model
    import innvestigate.utils as iutils
    print(ppl.files[0].folder)
    b = find_best_models(ppl.files[0].folder).val_acc.path
    model = load_model(str(b))
    model.summary()

    # rename layers, because we get layer name not unique error when loading partial model
    for l in model.layers:
        l._name = f'my_{l.name}'
    #layer = 'my_max_pooling2d_3'

    # Only the partial model is needed for the visualizers. Use innvestigate.utils.keras.graph.pre_softmax_tensors()
    partial_model = Model(
        inputs=model.inputs,
        outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs),
        name=model.name,
    )

    mp = ppl.task.get_model_task().mp
    get_kfold_set(ppl).test.run()
    x, y, _ = get_kfold_set(ppl).test.preproc_from(ppl).get_xyw(mp)
    every_n = math.ceil(x.shape[0]/(len(mp.num_label)*5))
    x, y = x[::every_n], y[::every_n]

    if selection is not None:
        x, y = x[selection], y[selection]

    # calc images per row
    images_per_row = max(2, math.ceil(800/(x.bounding_shape(1) if isinstance(x, RaggedTensor) else x.shape[1])))
    print('images_per_row:', images_per_row)
    print('test dataset shape:', x.bounding_shape() if isinstance(x, RaggedTensor) else x.shape)

    sample_count = (x.bounding_shape(0) if isinstance(x, RaggedTensor) else x.shape[0])
    x_size = min(sample_count, images_per_row)
    y_size = math.ceil(sample_count/images_per_row)

    fig, axes = plt.subplots(nrows=y_size*2, ncols=x_size, figsize=(14, y_size*6))
    for ax in axes.ravel().tolist():
        ax.set_xticks([]), ax.set_yticks([])

    for i, (x, y) in enumerate(zip(x, y)):
        if isinstance(x, RaggedTensor):
            x = x.numpy()

        ty = y.argmax()
        x_pos = i % images_per_row
        y_pos = (math.floor(i/images_per_row) * 2)

        inx = x.reshape((1, *x.shape))
        py = model.predict(inx).argmax()

        title = f'{mp.num_label[ty]}' + (f' ({i})' if selection is None else '')
        axes[y_pos][x_pos].set_title(title)
        axes[y_pos][x_pos].imshow(x.reshape(x.shape[:2]).T, aspect='auto', cmap='afmhot_r', interpolation=None)

        lrp_analyzer = LRP(partial_model, target_id=ty, relu=True, low=-1, high=1)
        analysis_lrp = lrp_analyzer.analyze(inx)[0]

        if len(analysis_lrp.shape) > 2:
            analysis_lrp = analysis_lrp.sum(axis=(2))

        M = np.abs(analysis_lrp).max()
        axes[y_pos+1][x_pos].imshow(analysis_lrp.T, vmin=-M, vmax=M, aspect='auto', cmap='seismic')

        for sp in ['top','bottom','left','right']:
            axes[y_pos+1][x_pos].spines[sp].set_linewidth(3)
            axes[y_pos+1][x_pos].spines[sp].set_color('green' if py == ty else 'red')
    fig.tight_layout()
    return fig


def create_sorted_result_set(model_name):
    result_file = BSC_ROOT_DATA_FOLDER / 'results' / f'{model_name}.json'
    results = json.loads(read_file(result_file))['results']
    return sorted(results.values(), key=operator.itemgetter('test_acc_m', 'val_acc_m'), reverse=True)


def create_model_overview(sorted_results):
    data = {}
    keys = list(map("".join, itertools.product(*[['val', 'test'], ['_acc', '_loss'], ['_m', '_d']])))
    for result in sorted_results:
        data[result['id']] = {key: result[key] for key in keys}

    return pd.DataFrame().from_dict(data, orient="index")


def create_info_cell(result_model: ReportResult, result_name):
    template = RESULTS_FOLDER / 'report_template' / f'{result_name}.md'
    if not template.exists():
        template = RESULTS_FOLDER / 'report_template' / 'default.md'

    display(Markdown(parse_template(read_file(template), result_model)))


def create_statistics(plt, sorted_results, result_model: ReportResult):
    model_path = result_model.get_model_path()

    for result in sorted_results:
        id = result['id']
        model_kFold_path = model_path / id
        statFiles = model_kFold_path.parent.parent.glob('*/*.stats.json')

        for statFile in statFiles:
            data = json.loads(read_file(statFile))

            if len(data) == 0: continue

            localStatFile = str(statFile)[len(str(MODELS_FOLDER)) + 1:]
            display(HTML('<h3>Result for: {0:s}<h3>'.format("/".join(localStatFile.split(os.sep)[0:2]))))
            display(localStatFile)
            fig, axes = split_acc_loss_plots(plt, range(1, len(data) + 1),
                                             [h['acc'] for h in data],
                                             [h['val_acc'] for h in data],
                                             [h['loss'] for h in data],
                                             [h['val_loss'] for h in data])
            display(fig)
            fig.clear()
            plt.close(fig)

            reports = [json.loads(read_file(f)) for f in Path(statFile).parent.glob('rec/*.report.json')]
            for report in reports:
                display(report['id'])
                display(pd.DataFrame(report['confusion']))


def create_spectrogram_overview(image_paths, labels, bin_size):
    for label in labels:
        df = pd.DataFrame(columns=['A', 'B', 'C', 'D', 'E'])
        filtered_image_paths = list(filter(lambda image_path: image_path.find('_' + label + '.') != -1, image_paths))
        all = len(filtered_image_paths)

        display(
            label + ', all:{0}, train:{1}, val:{2}, test:{3}, reminder:{4}'.format(
                all, all - (3 * bin_size), bin_size, bin_size * 2, all - (8 * bin_size)
            )
        )

        random_filtered_image_paths = random.sample(filtered_image_paths, 20)
        for row in range(4):
            df.loc[row] = [
                '<img title="{0:s}" src="data:image/png;base64,{1:s}">'.format(
                    p[len(str(BSC_ROOT_DATA_FOLDER)) + 1:]
                    , base64.b64encode(Image(filename=p).data).decode()
                )
                for p in random_filtered_image_paths[row * 5:row * 5 + 5]
            ]

        display(HTML(df.to_html(escape=False)))


def get_best_model_path_from_result(result):
    recognize_best_val_acc = list(filter(lambda action: action['task'] == 'ParallelTask' and 'action' in action['props'] and action['props']['action'] == 'recognize_best_val_acc', result['pipeline']))
    model_path = recognize_best_val_acc[0]['props']['kwargs']['model_path']
    assert str(BSC_DATA_FOLDER).endswith(model_path.split('/')[0]), f'invalid data folder, {model_path.split("/")[0]} is required'
    return expand_data_path(model_path)


def get_best_model_pipeline_from_result(result):
    for file in get_best_model_path_from_result(result).parent.glob('epoch_*.h5.json'):
        return import_pipeline_file(file)

    print(json.dumps(result, indent=2))
    raise Exception('could not find best model')


def get_best_result_for_model(result_name, model):
    return list(filter(lambda result: model in result['id'], create_sorted_result_set(result_name)))[0]
