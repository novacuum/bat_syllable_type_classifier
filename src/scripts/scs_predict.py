from engine.audio import load_audio_matching_files
from engine.helpers import read_file
from engine.nn.training import NNModel
from engine.settings import BSC_ROOT_DATA_FOLDER
from utils.report import get_best_result_for_model, get_best_model_pipeline_from_result


model: NNModel = get_best_model_pipeline_from_result(get_best_result_for_model('testing_scs_r3', 'nn_densNet'))
test_audio = list(map(lambda f: str(f), (BSC_ROOT_DATA_FOLDER / 'audio' / 'prediction').glob('*.wav')))

for prediction_csv in model.predict_best_val_acc(load_audio_matching_files(test_audio)).run().files:
    print(prediction_csv.path())
    print(read_file(prediction_csv.path()))
