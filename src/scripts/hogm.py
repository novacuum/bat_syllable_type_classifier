from engine.export import load_audio, metadata_db
from scripts.models.datasets import srne1_c2_base

mdb = metadata_db('metadata.json')

"""PS5 best model"""

base = (
    srne1_c2_base().multi().hog_hwr_features()
)


def run(s, m, dataset):
    return base \
        .create_hmm_model(name='hogm', states=s) \
        .train(mixtures=m) \
        .recognize(load_audio(mdb, dataset)) \
        .extract('woodcock-dec23') \
        .get()['accuracy']


run(5, 17, 'validation.txt')
run(5, 19, 'validation.txt')
run(5, 19, 'testing.txt')
