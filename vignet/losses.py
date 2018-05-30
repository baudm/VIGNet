#!/usr/bin/env python3

from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective


def image_loss(y_true, y_pred):
    dssim = DSSIMObjective()
    return dssim(y_true, y_pred) + mean_squared_error(y_true, y_pred)
