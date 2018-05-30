import sys

import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling3D, LeakyReLU, Conv3D, \
    Activation, Flatten, Dense, Reshape, BatchNormalization, Lambda
from keras.optimizers import Adam

from vignet.rnn3D import LSTM3D
from vignet.voxel import voxel2obj
from vignet.vox2mesh_func import vox2mesh


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


def create_3d_autoencoder(vox_size=32):
    input_shape = (128, 128, 4)
    inp = Input(shape=input_shape, name='input_image')

    # Denormalize image from [-1, 1] to [0, 255]
    x = Lambda(lambda i: (i + 1.) * 127.5)(inp)

    x = Conv2D(96, (7, 7), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    input_conv_lstm3d_shape = (1, 1024)
    x = Reshape(target_shape=input_conv_lstm3d_shape)(x)

    rnn_layer = LSTM3D(128)(x)

    x = UpSampling3D((2, 2, 2))(rnn_layer)

    x = Conv3D(128, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = UpSampling3D((2, 2, 2))(x)

    x = Conv3D(128, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = UpSampling3D((2, 2, 2))(x)

    x = Conv3D(64, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if vox_size >= 64:
        x = UpSampling3D((2, 2, 2))(x)

    x = Conv3D(32, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    if vox_size >= 128:
        x = UpSampling3D((2, 2, 2))(x)

    x = Conv3D(2, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    # if vox_size >= 256:
    # 	x = UpSampling3D((2,2,2))(x)

    out = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=out, name='voxelizer')
    return model


if __name__=='__main__':
    print("STARTING...")

    # GET TEST DATA
    print("[1] FETCHING DATA...")
    # VARIABLE 'im' IS BLANK
    input = np.asarray(Image.open(sys.argv[1], 'r'))

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
