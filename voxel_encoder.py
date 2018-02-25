import numpy as np
import random as random
import keras.backend as K
from keras.utils import plot_model

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling3D, LeakyReLU, Conv3D, Add, Subtract, Multiply, Activation, Flatten, Dense, Reshape, Layer
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.engine.topology import Layer

import tensorflow as tf
import gc, sys, glob

from voxel import voxel2obj
from vox2mesh_func import vox2mesh
from PIL import Image
import cv2

# Loss
def shapeloss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    dy_pred = K.clip((1-y_pred), K.epsilon(), 1)
    l = -K.mean(y_true*K.log(y_pred) + (1 - y_true)*K.log(dy_pred))
    return l

# Jaccard Distance
def voxel_err(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def preprocess_before_feed(im):
    im = Image.fromarray(im.astype('uint8'), 'RGBA')
    orig = im

    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", im.size, (255,255,255,255))
        bg.paste(im, mask=alpha)
        #print('TRANSPARENCY REMOVED')

    im = cv2.equalizeHist(cv2.cvtColor(cv2.resize(np.array(im), (127, 127), interpolation = cv2.INTER_NEAREST), cv2.COLOR_BGR2GRAY))
    im = cv2.resize(np.array(bg), (127, 127), interpolation = cv2.INTER_NEAREST)
    im = cv2.resize(np.array(im), (127, 127), interpolation = cv2.INTER_NEAREST)
    im_fin = np.zeros((127,127,3));
    im_fin[:,:,0] = im[:,:,0]
    im_fin[:,:,1] = im[:,:,1]
    im_fin[:,:,2] = im[:,:,2]
    im = im_fin

    return im


def denormalize_image(x):
    x = (x + 1.) / 2.
    x = np.clip(x, 0., 1.)
    x = x * 255
    return x


def preprocess_batch(batch_im):
    batch_im = denormalize_image(batch_im)
    batch_proc = []
    for im in batch_im:
        proc = preprocess_before_feed(im)
        batch_proc.append(proc)
    return np.stack(batch_proc).astype('float32') / 255.


class Preprocessor(Layer):

    def call(self, inputs, **kwargs):
        y = tf.py_func(preprocess_batch, [inputs], [tf.float32])
        y[0].set_shape((None, 127, 127, 3))
        return y

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = 3
        output_shape[-2] = 127
        output_shape[-3] = 127
        return tuple(output_shape)


def create_3d_autoencoder():
    input_shape = (128,128,4)
    inp = Input(shape=input_shape, name='input_image')

    proc_inp = Preprocessor(name='preprocessor')(inp)

    conv1a = Conv2D(96, (7,7), strides=(1, 1), padding='same', name='conv1a')(proc_inp)
    rect1a = Activation('relu')(conv1a)
    conv1b = Conv2D(96, (3,3), strides=(1, 1), padding='same', name='conv1b')(rect1a)
    rect1 = Activation('relu')(conv1b)
    pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(rect1)

    conv2a = Conv2D(128, (3,3), strides=(1, 1), padding='same', name='conv2a')(pool1)
    rect2a = Activation('relu')(conv2a)
    conv2b = Conv2D(128, (3,3), strides=(1, 1), padding='same', name='conv2b')(rect2a)
    rect2 = Activation('relu')(conv2b)
    conv2c = Conv2D(128, (1,1), strides=(1, 1), padding='same', name='conv2c')(pool1)
    res2 = Add()([conv2c, rect2])
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(res2)

    conv3a = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv3a')(pool2)
    rect3a = Activation('relu')(conv3a)
    conv3b = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv3b')(rect3a)
    rect3 = Activation('relu')(conv3b)
    conv3c = Conv2D(256, (1,1), strides=(1, 1), padding='same', name='conv3c')(pool2)
    res3 = Add()([conv3c, rect3])
    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(res3)

    conv4a = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv4a')(pool3)
    rect4a = Activation('relu')(conv4a)
    conv4b = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv4b')(rect4a)
    rect4 = Activation('relu')(conv4b)
    pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(rect4)

    conv5a = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv5a')(pool4)
    rect5a = Activation('relu')(conv5a)
    conv5b = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv5b')(rect5a)
    rect5 = Activation('relu')(conv5b)
    conv5c = Conv2D(256, (1,1), strides=(1, 1), padding='same', name='conv5c')(pool4)
    res5 = Add()([conv5c, rect5])
    pool5 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(res5)

    conv6a = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv6a')(pool5)
    rect6a = Activation('relu')(conv6a)
    conv6b = Conv2D(256, (3,3), strides=(1, 1), padding='same', name='conv6b')(rect6a)
    rect6 = Activation('relu')(conv6b)
    res6 = Add()([pool5, rect6])
    pool6 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(res6)

    flat6 = Flatten()(pool6)
    fc7 = Dense(1024, name='dense')(flat6)
    rect7 = Activation('relu')(fc7)

    # Dummy 3D grid hidden representations
    prev_s = Reshape((2,2,2,128))(rect7)
    t_x_rs = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='bottleneck')(prev_s)

    ## DECODER
    unpool7 = UpSampling3D(size=(2,2,2))(t_x_rs)
    conv7a = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv7a')(unpool7)
    rect7a = Activation('relu')(conv7a)
    conv7b = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv7b')(rect7a)
    rect7 = Activation('relu')(conv7b)
    res7 = Add()([conv7a, rect7])

    unpool8 = UpSampling3D(size=(2,2,2))(res7)
    conv8a = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv8a')(unpool8)
    rect8a = Activation('relu')(conv8a)
    conv8b = Conv3D(128, (3,3,3), strides=(1,1,1), padding='same', name='conv8b')(rect8a)
    rect8 = Activation('relu')(conv8b)
    res8 = Add()([conv8a, rect8])

    unpool9 = UpSampling3D(size=(2,2,2))(res8)
    conv9a = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', name='conv9a')(unpool9)
    rect9a = Activation('relu')(conv9a)
    conv9b = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', name='conv9b')(rect9a)
    rect9 = Activation('relu')(conv9b)

    conv9c = Conv3D(64, (1,1,1), strides=(1,1,1), padding='same', name='conv9c')(unpool9)
    res9 = Add()([conv9c, rect9])

    unpool10 = UpSampling3D(size=(2,2,2))(res9)
    conv10a = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', name='conv10a')(unpool10)
    rect10a = Activation('relu')(conv10a)
    conv10b = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', name='conv10b')(rect10a)
    rect10 = Activation('relu')(conv10b)

    conv10c = Conv3D(32, (3,3,3), strides=(1,1,1), padding='same', name='conv10c')(rect10a)
    res10 = Add()([conv10c, rect10])

    conv11 = Conv3D(2, (3,3,3), strides=(1,1,1), padding='same', name='conv11')(res10)
    out = Activation('softmax')(conv11)

    model = Model(inputs =inp, outputs=out, name='voxelizer')
    return model


if __name__=='__main__':
    print("STARTING...")
    for i in range(20):
        gc.collect()

    # GET TEST DATA
    print("[1] FETCHING DATA...")
    # VARIABLE 'im' IS BLANK
    input = preprocess_before_feed(im)

    # load json and create model
    print("[2] LOADING MODEL...")
    # loaded_model.load
    loaded_model = create_3d_autoencoder()
    loaded_model.summary()
    # load weights into new model
    loaded_model.load_weights("model_vignet_santelices.h5")
    print("Loaded model from disk")
    # Re-compile
    loaded_model.compile(optimizer=Adam(lr=0.0001), loss=shapeloss, metrics=[voxel_err])

    print("[3] EVALUATION OF MODEL ON TEST DATA...")
    # score = loaded_model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose = 1)
    # print('%s: %0.2f' % (loaded_model.metrics_names[0], score[0]))
    # print('%s: %0.2f' % (loaded_model.metrics_names[1], score[1]))

    # Testing
    voxel_prediction = loaded_model.predict(np.expand_dims(input, axis=0), verbose=0)[0]

    # Save prediction into a file named 'prediction.obj' or the given argument
    # pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'
    result = Image.fromarray((input * 255).astype(np.uint8))
    result.save('out.png')
    # Save the prediction to an OBJ file (mesh file)
    vox2mesh('prediction_mesh.obj', voxel_prediction[:, :, :, 1] > 0.4)
    voxel2obj('prediction_vox.obj', voxel_prediction[:, :, :, 1] > 0.4)