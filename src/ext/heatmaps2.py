import numpy as np
from tensorflow.keras import backend as K
from innvestigate.analyzer import BoundedDeepTaylor
from tensorflow.keras.layers import Lambda


class _MaskedDeepTaylor(BoundedDeepTaylor):
    """Give any specific path to the DTD
    """

    def __init__(self, model, R_mask, **kwargs):
        super(_MaskedDeepTaylor, self).__init__(
            # model, neuron_selection_mode="all", **kwargs)
            model, **kwargs)
        self.initialize_r_mask(R_mask)

    def initialize_r_mask(self, R_mask):
        """Mask R road
        Arguments:
            initial_R_mask {[type]} -- [description]
        """

        self.R_mask = K.constant(R_mask)

    def _head_mapping(self, X):
        """Multiplication with the initialized one-hot vector
        """
        initial_R = Lambda(lambda x: (x * self.R_mask))(X)
        return initial_R


class LRP(_MaskedDeepTaylor):
    def __init__(self,
                 model,
                 target_id,
                 relu=False,
                 low=-1.,
                 high=1.,
                 **kwargs):
        """Target value:same as prediction，otherwise:0
        Arguments:
            model {[type]} -- [description]
            target_id {[type]} -- [description]
            predictions {[type]} -- [description]
        """
        self.relu = relu
        R_mask = np.zeros((model.output_shape[1]))
        R_mask[target_id] = 1
        super(LRP, self).__init__(model, R_mask=R_mask, low=low, high=high, **kwargs)

    def analyze(self, inputs):
        if self.relu:
            x = super(LRP, self).analyze(inputs)
            print('lrp, returned layers', x.keys())
            return np.maximum(next(iter(x.values())), 0)
        else:
            return super(LRP, self).analyze(inputs)
