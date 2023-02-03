import matplotlib.pyplot as plt


def create_color_map_for_symbols(symbols, cmap='viridis'):
    color_map = plt.cm.get_cmap(cmap, len(symbols))
    return {symbol: color_map(i) for i, symbol in enumerate(symbols)}


LABELS = ['B2', 'B3', 'B4', 'VS', 'VSV', 'UPS', 'none']
LABEL_COLORS = create_color_map_for_symbols(LABELS)  # cubehelix


MODELS = ['denseNet', 'cnn_2d', 'cnn_1d', 'lstm']
MODEL_COLORS = create_color_map_for_symbols(MODELS)
MODEL_NAMES = ['DenseNet', 'CNN 2D', 'CNN 1D', 'LSTM']
MODEL_NAMES_LONG = ['DenseNet 121 Model', 'CNN Model with 2D convolution', 'CNN Model with 1D convolution', 'LSTM Model']




