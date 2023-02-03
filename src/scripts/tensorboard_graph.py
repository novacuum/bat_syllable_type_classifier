from engine.settings import LOG_FOLDER
from scripts.models.cnn1d_model import create_cnn_1d_model

log_dir = LOG_FOLDER / 'tensorboard'

"""
Helper for create graph log/json files for tensorboard
tensorboard --logdir data/log/tensorboard
"""


def existing_model(result_name, model):
    import tensorflow as tf
    from tensorflow.python.keras.models import load_model
    from tensorflow.python.ops import summary_ops_v2
    from utils.report import get_best_model_path_from_result, get_best_result_for_model

    writer = tf.summary.create_file_writer(str(log_dir))
    with writer.as_default():
        model_path = get_best_model_path_from_result(get_best_result_for_model(result_name, model))
        model = load_model(str(model_path))
        summary_ops_v2.keras_model('keras', model, step=0)
    writer.close()


def new_model():
    import numpy as np
    from tensorflow.python.keras.callbacks import TensorBoard
    from scripts.models.lstm_model import create_lstm_model
    num_labels = 5
    shape = (20, 400, 256)
    # model = create_lstm_model(512, num_labels, 0.0001)
    model = create_cnn_1d_model(shape[1:], 6, 0.001)
    return
    tb_cbk = TensorBoard(str(log_dir), write_graph=True)
    x, y = np.ones(shape), np.ones((20, num_labels))
    model.fit(x, y, batch_size=5, epochs=2, callbacks=[tb_cbk])


# existing_model('testing_sct_compressed', 'nn_densNet')
new_model()

print('for stating the tensorboard, run:')
print('tensorboard --logdir ' + str(log_dir))
