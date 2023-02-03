from utils.report import nn_cnn_2d_lrp
from IPython.display import display
from engine.settings import BSC_ROOT_DATA_FOLDER
from matplotlib import pyplot as plt


# simple_call_test/models/nn_densNet_sct_padded_nrs0xpps2000h256_raw_100/dac454f20a7dcfbed0d3c7694888ef2acfc1ebd0/
# fig = nn_cnn_2d_lrp(BSC_ROOT_DATA_FOLDER / 'simple_call_test' / 'models' / 'nn_densNet_sct_padded_nrs0xpps2000h256_raw_100' / 'dac454f20a7dcfbed0d3c7694888ef2acfc1ebd0' / 'epoch_100.h5.json')

#simple_call_test/models/nn_cnn_2d_sct_compressed_nrs0_raw_100/72304d836ad5d2c6e3bc183d21bb4aac1bfd9db6/best_val_acc_e024_v0.95.h5
# fig = nn_cnn_2d_lrp(BSC_ROOT_DATA_FOLDER / 'simple_call_test' / 'models' / 'nn_cnn_2d_sct_compressed_nrs0_raw_100' / '72304d836ad5d2c6e3bc183d21bb4aac1bfd9db6' / 'epoch_100.h5.json')
fig = nn_cnn_2d_lrp(BSC_ROOT_DATA_FOLDER / 'simple_call_test' / 'models' / 'nn_densNet_sct_compressed_nrs0_raw_100' / 'c705a68d6b660af11476f6d6e76df7837f882209' / 'epoch_100.h5.json')
fig.savefig('./test.png', dpi=300)
fig.clear()
plt.close(fig)
