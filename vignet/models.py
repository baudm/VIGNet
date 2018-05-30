#!/usr/bin/env python3

from keras import layers, models

from . import voxel_encoder
from .layers import PrimaryCap, CapsuleLayer, Mask, identity_block, conv_block, conv2d_transpose_bn


def build_image_decoder(n_class, capsule_size):
    masked_dcaps = layers.Input((n_class, capsule_size), name='masked_digitcaps')
    conv = layers.Input((112, 112, 256), name='conv1')

    y = layers.Reshape((1, 1, n_class*capsule_size))(masked_dcaps)

    y = layers.Conv2DTranspose(32, 7, name='conv_pre')(y)

    y = identity_block(y, 3, 32, stage='1', block='a')
    y = identity_block(y, 3, 32, stage='1', block='b')

    y = conv_block(y, 3, 64, stage='2', block='a')
    y = identity_block(y, 3, 64, stage='2', block='b')
    y = identity_block(y, 3, 64, stage='2', block='c')

    y = conv_block(y, 3, 128, stage='3', block='a')
    y = identity_block(y, 3, 128, stage='3', block='b')
    y = identity_block(y, 3, 128, stage='3', block='c')

    y = conv_block(y, 3, 256, stage='4', block='a')
    y = identity_block(y, 3, 256, stage='4', block='b')
    y = identity_block(y, 3, 256, stage='4', block='c')

    y = conv_block(y, 3, 256, stage='5', block='a')

    y = layers.concatenate([y, conv])

    y = identity_block(y, 3, 512, stage='6', block='a')
    y = identity_block(y, 3, 512, stage='6', block='b')
    y = identity_block(y, 3, 512, stage='6', block='c')

    y = conv_block(y, 3, 128, stage='7', block='a', strides=1)
    y = identity_block(y, 3, 128, stage='7', block='b')
    y = identity_block(y, 3, 128, stage='7', block='c')

    y = conv2d_transpose_bn(y, 4, 17, activation='tanh', name='out_recon')

    model = models.Model([conv, masked_dcaps], y, name='image_decoder')
    return model


def build_pose_decoder(n_class, capsule_size):
    masked_dcaps = layers.Input((n_class, capsule_size), name='masked_digitcaps')
    pose = layers.Flatten()(masked_dcaps)
    pose = layers.Dense(512, activation='relu', kernel_initializer='he_uniform', name='dense_1')(pose)
    pose = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform', name='dense_2')(pose)
    pose = layers.Dense(3, activation='tanh', name='pose_output')(pose)
    model = models.Model(masked_dcaps, pose, name='pose_decoder')
    return model


def VIGNet(input_shape, n_class, routings, capsule_size=16):
    """
    A Capsule Network on ShapeNet.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, name='input_image')

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=17, strides=1, padding='valid', name='conv1', kernel_initializer='he_uniform')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=16, strides=3, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=capsule_size, routings=routings,
                             name='digitcaps')(primarycaps)

    # Decoder network.
    y1 = layers.Input(shape=(n_class,), name='input_label_1')
    y2 = layers.Input(shape=(n_class,), name='input_label_2')
    masked_by_y1 = Mask(name='masked_digitcap_1')([digitcaps, y1])  # The true label is used to mask the output of capsule layer. For training
    masked_by_y2 = Mask(name='masked_digitcap_2')([digitcaps, y2])  # The true label is used to mask the output of capsule layer. For training

    #masked1 = Mask(1)(digitcaps) # Mask using the capsule with maximal length. For prediction
    #masked2 = Mask(2)(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    image_decoder = build_image_decoder(n_class, capsule_size)


    pose_decoder = build_pose_decoder(n_class, capsule_size)


    voxelizer = voxel_encoder.create_3d_autoencoder()

    x1_pred = image_decoder([conv1, masked_by_y1])
    x2_pred = image_decoder([conv1, masked_by_y2])

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y1, y2], [x1_pred, x2_pred,
                                             pose_decoder(masked_by_y1), pose_decoder(masked_by_y2)])

    test_model = models.Model([x, y1, y2], [x1_pred, x2_pred,
                                             pose_decoder(masked_by_y1), pose_decoder(masked_by_y2),
                                             voxelizer(x1_pred), voxelizer(x2_pred)])

    return train_model, test_model, voxelizer
