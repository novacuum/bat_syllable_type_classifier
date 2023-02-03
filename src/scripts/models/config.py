import math

"""General configuration for the experiment scripts

Author: Gilles Waeber, VII 2019"""

MAX_DURATION = 9.18

IMG_PEAK_XPPS_1 = 56
IMG_PEAK_XPPS_2 = 200
IMG_PEAK_XPPS_3 = 400
IMG_PAD_XPPS = 56
IMG_PAD_MAX_WIDTH = int(math.ceil(IMG_PAD_XPPS * MAX_DURATION))
IMG_STRETCHED_WIDTH = 512

img_height = 256
learning_rate = .0001
max_epochs = 100
save_every = max_epochs

hog_num_bins = 12
lstm_neurons = 256
padding = 'pre'
