import json
import traceback
from multiprocessing import cpu_count
from multiprocessing.pool import Pool, ThreadPool

from engine.audio import load_audio
from engine.helpers import read_file
from engine.settings import RESULTS_FOLDER
from engine.utils import print_log
from scripts.models.datasets import Dataset

"""HMM extensive parameter search

Author: Gilles Waeber, VI 2019"""


THREADS = cpu_count() - 1


def _runner(f_t_v_ml_s_dn):
    f, t, val, mixtures_list, s, dataset_name = f_t_v_ml_s_dn
    try:
        model = t.create_hmm_model(f'{f}_{dataset_name}', states=s)
        for m in mixtures_list:
            model.train(mixtures=m).recognize(val).extract(f'hmm_{f}_{dataset_name}').run()
        return 1
    except:
        error = ''.join(traceback.format_exc())
        print_log(f'Exception for f={f}, s={s}:\n{error}', 'yellow')


def all_hmm_training_sets(base, *, img_width, img_xpps, img_height):
    return {
        'hog': base.create_spectrogram(height=img_height, width=img_width).hog_hwr_features().run(),
        'raw': base.create_spectrogram(height=img_height, width=img_width).img_features().run(),
        'hog_m': base.create_spectrogram(height=img_height, width=img_width).hog_hwr_features().multi().run(),
        'raw_m': base.create_spectrogram(height=img_height, width=img_width).img_features().multi().run(),
        'hog_var': base.create_spectrogram(height=img_height, x_pixels_per_sec=img_xpps).hog_hwr_features().run(),
        'raw_var': base.create_spectrogram(height=img_height, x_pixels_per_sec=img_xpps).img_features().run(),
        'hog_m_var': base.create_spectrogram(height=img_height,
                                             x_pixels_per_sec=img_xpps).hog_hwr_features().multi().run(),
        'raw_m_var': base.create_spectrogram(height=img_height, x_pixels_per_sec=img_xpps).img_features().multi().run(),
    }


def get_states_mixtures(struct):
    for p in struct['pipeline']:
        if p['task'] == 'CreateModelTask':
            s = p['props']['states']
            break
    for p in struct['pipeline']:
        if p['task'] == 'TrainModelTask':
            m = p['props']['mixtures']
            break
    return s, m


def test_hmm_model_k_fold(
    *,
    dataset: Dataset,
    mixtures,
    states,
    base,
    model_suffix,
    pool: Pool
):
    model = base.create_hmm_model(f'{dataset.name}{model_suffix}', states=states)
    best = model.evaluate_model(mixtures=mixtures).recognize_best_val_acc(parallel=pool)
    best.extract(f'testing_{dataset.name}').run(parallel=pool)


def test_hmm_model(
    *,
    dataset: Dataset,
    mixtures_list,
    states_list,
    validation_file,
    testing_file,
    training_sets,
    mdb,
):
    with ThreadPool(THREADS) as p:
        ftv = [(f, t, load_audio(mdb, validation_file).preproc_from(t).run()) for f, t in
               training_sets.items()]

        p.map(_runner, [(f, t, val, mixtures_list, s, dataset.name)
                        for s in reversed(states_list)
                        for f, t, val in ftv])

    for f, t in training_sets.items():
        reports = json.loads(read_file(f'{RESULTS_FOLDER}/hmm_{f}_{dataset.name}.json'))['results']
        best = sorted(reports.values(), key=lambda r: (
            -r['accuracy'],
            *get_states_mixtures(r)
        ))[0]
        states, mixtures = get_states_mixtures(best)

        print(f'Best accuracy for {f}: {best["accuracy"]:.2%} with {states} states and {mixtures} mixtures '
              f'({best["correct"]}/{best["total"]})')

        best_model = t.create_hmm_model(f'{f}_{dataset.name}', states=states).train(mixtures=mixtures)
        best_model.recognize(load_audio(mdb, testing_file)).extract(f'testing_{dataset.name}').run()
