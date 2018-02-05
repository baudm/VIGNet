"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask, k_categorical_accuracy
from combine_mnist import sample_and_combine
from buffering import buffered_gen_threaded as buf

K.set_image_data_format('channels_last')

from keras.utils.vis_utils import plot_model


def conv2d_bn(x, filters, kernel_size, strides=(1, 1), padding='valid', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def conv2d_transpose_bn(x, filters, kernel_size, strides=(1, 1), padding='valid', activation='relu'):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def res_conv2d_transpose_bn(x, filters, kernel_size, strides=(1, 1), padding='valid', activation='relu'):
    # block 1
    res = layers.BatchNormalization()(x)
    res = layers.Activation(activation)(res)
    res = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(res)
    # block 2
    res = layers.BatchNormalization()(res)
    res = layers.Activation(activation)(res)
    res = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(res)
    return layers.add([x, res])



from keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    filters_bottleneck = filters // 4

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters_bottleneck, (1, 1), padding='same', name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters_bottleneck, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters, (1, 1), padding='same', name=conv_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    filters_bottleneck = filters // 4

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    shortcut = Conv2DTranspose(filters, (1, 1), strides=strides, padding='same',
                      name=conv_name_base + '1')(x)

    x = Conv2DTranspose(filters_bottleneck, (1, 1), strides=strides, padding='same',
               name=conv_name_base + '2a')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters_bottleneck, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(filters, (1, 1), padding='same', name=conv_name_base + '2c')(x)

    x = layers.add([x, shortcut])

    return x


def CapsNet(input_shape, decoder_output_shape, n_class, routings, capsule_size=16):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=17, strides=1, padding='valid', name='conv1')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=16, strides=3, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=capsule_size, routings=routings,
                             name='digitcaps')(primarycaps)

    def res_decoder():
        dcaps = layers.Input((n_class, capsule_size))
        # pcaps = layers.Input((100352, 8))
        conv = layers.Input((112, 112, 256))

        y = layers.Reshape((1, 1, n_class*capsule_size))(dcaps)

        y = Conv2DTranspose(32, 7)(y)
        # y = BatchNormalization()(y)
        # y = Activation('relu')(y)

        y = identity_block(y, 3, 32, '1', '1')
        y = identity_block(y, 3, 32, '1', '2')
        y = conv_block(y, 3, 64, '1', '3')

        y = identity_block(y, 3, 64, '2', '1')
        y = identity_block(y, 3, 64, '2', '2')
        # y = identity_block(y, 3, 64, '2', '3')
        y = conv_block(y, 3, 128, '2', '4')

        y = identity_block(y, 3, 128, '3', '1')
        y = identity_block(y, 3, 128, '3', '2')
        # y = identity_block(y, 3, 128, '3', '3')
        # y = identity_block(y, 3, 128, '3', '4')
        # y = identity_block(y, 3, 128, '3', '5')
        y = conv_block(y, 3, 256, '3', '6')

        y = identity_block(y, 3, 256, '4', '1')
        y = identity_block(y, 3, 256, '4', '2')
        # y = identity_block(y, 3, 256, '4', '3')
        y = conv_block(y, 3, 256, '4', '4')

        y = layers.concatenate([y, conv])

        y = identity_block(y, 3, 512, '5', '1')
        y = identity_block(y, 3, 512, '5', '2')
        y = identity_block(y, 3, 512, '5', '3')

        y = conv_block(y, 3, 128, '5', '4', strides=1)

        y = identity_block(y, 3, 128, '6', '2')
        y = identity_block(y, 3, 128, '6', '3')

        y = conv2d_transpose_bn(y, 4, 17, activation='tanh')
        # y = identity_block(y, 3, 256, '6', '1')
        # y = identity_block(y, 3, 256, '6', '2')

        # y = conv_block(y, 3, 512, '5', '4', strides=1)
        #
        # y = identity_block(y, 3, 512, '6', '1')
        # y = identity_block(y, 3, 512, '6', '2')
        # y = identity_block(y, 3, 512, '6', '3')
        # y = conv_block(y, 3, 512, '6', '4', strides=1)

        # y = Conv2DTranspose(3, 1, padding='same')(y)
        # y = BatchNormalization()(y)
        #
        #
        # # y = Conv2DTranspose(128, 5)(y)
        # # y = BatchNormalization()(y)
        # # y = Activation('relu')(y)
        # #
        # # y = Conv2DTranspose(256, 5)(y)
        # # y = BatchNormalization()(y)
        # # y = Activation('relu')(y)
        # #
        # # # y = layers.concatenate([y, conv])
        # #
        # # y = Conv2DTranspose(64, 5)(y)
        # # y = BatchNormalization()(y)
        # # y = Activation('relu')(y)
        # #
        # # y = Conv2DTranspose(3, 5)(y)
        # # y = BatchNormalization()(y)
        #
        #
        # y = layers.Activation('sigmoid', name='out_recon')(y)

        m = models.Model([conv, dcaps], y, name='decoder')
        # m.summary()
        # plot_model(m, show_shapes=True, to_file='decoder.png')
        return m


    #
    # m = models.Model(x, y)
    # m.summary()
    # plot_model(m, show_shapes=True)
    # return

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y1 = layers.Input(shape=(n_class,))
    y2 = layers.Input(shape=(n_class,))
    masked_by_y1 = Mask()([digitcaps, y1])  # The true label is used to mask the output of capsule layer. For training
    masked_by_y2 = Mask()([digitcaps, y2])  # The true label is used to mask the output of capsule layer. For training
    masked1 = Mask(1)(digitcaps) # Mask using the capsule with maximal length. For prediction
    masked2 = Mask(2)(digitcaps)  # Mask using the capsule with maximal length. For prediction



    # Shared Decoder model in training and prediction
    decoder = res_decoder()

    def make_pose_estimator():
        masked_dcaps = layers.Input((n_class, capsule_size))
        pose = layers.Flatten()(masked_dcaps)
        pose = layers.Dense(512, activation='relu')(pose)
        pose = layers.Dense(1024, activation='relu')(pose)
        pose = layers.Dense(3, activation='sigmoid', name='prepose')(pose)
        pose = layers.Reshape(target_shape=(3,), name='pose')(pose)
        m = models.Model(masked_dcaps, pose, name='pose')
        return m

    pose_estimator = make_pose_estimator()

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y1, y2], [out_caps, decoder([conv1, masked_by_y1]), decoder([conv1, masked_by_y2]), pose_estimator(masked_by_y1), pose_estimator(masked_by_y2)])
    eval_model = models.Model([x, y1, y2], [out_caps, decoder([conv1, masked_by_y1]), decoder([conv1, masked_by_y2]), pose_estimator(masked_by_y1), pose_estimator(masked_by_y2)])
    plot_model(train_model, show_shapes=True)

    # return None

    # manipulate model
    manipulate_model = train_model
    # noise = layers.Input(shape=(n_class, 16))
    # noised_digitcaps = layers.Add()([digitcaps, noise])
    # masked_noised_y = Mask()([noised_digitcaps, y1])
    # manipulate_model = models.Model([x, y1, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective

def image_loss(y_true, y_pred):
    dssim = DSSIMObjective()
    return dssim(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def preprocess_pose(pose):
    pose[:, 0] = (pose[:, 0] + (np.pi / 6.)) * 3. / np.pi
    # azimuth from 0 to 360deg
    pose[:, 1] = pose[:, 1] / (2. * np.pi)
    # distance factor from 1.0 to sqrt(2.0)
    pose[:, 2] = (pose[:, 2] - 1.) / (np.sqrt(2.) - 1.)


def data_generator(x_data, y_data, batch_size, overlap, test=False):
    while True:
        data = [sample_and_combine(x_data, y_data, overlap) for i in range(batch_size)]
        # Group
        data = zip(*data)
        # Stack
        data = list(map(np.stack, data))
        x1, x2, x, y1, y2, y, pose1, pose2 = data
        preprocess_pose(pose1)
        preprocess_pose(pose2)
        inputs = x if test else [x, y1, y2]
        yield inputs, [y, x1, x2, pose1, pose2]


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='decoder_loss_1',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, image_loss, image_loss, 'mse', 'mse'],
                  # loss_weights=[1., args.lam_recon, args.lam_recon],
                  metrics={'capsnet': k_categorical_accuracy})

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=buf(data_generator(x_train, y_train, args.batch_size, args.overlap), 3),
                        steps_per_epoch=1000,
                        epochs=args.epochs,
                        initial_epoch=args.initial_epoch,
                        # validation_data=buf(data_generator(x_test, y_test, args.batch_size, args.overlap), 3),
                        # validation_steps=500,
                        callbacks=[log, tb, checkpoint])#, lr_decay])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=100)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()




def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation, also known as the intersection-over-union loss.

    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.

    For example, assume you are trying to predict if each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?

    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # References

    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.

    https://en.wikipedia.org/wiki/Jaccard_index

    """
    intersection = np.sum(np.abs(y_true * y_pred), axis=-1)
    sum_ = np.sum(np.abs(y_true) + np.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def test_multi(model, data, args):
    x_test, y_test = data
    x1, x2, x, y1, y2, y = sample_and_combine(x_test, y_test, overlap_factor=args.overlap)
    y_pred, x_recon1, x_recon2 = model.predict_on_batch([x[np.newaxis], y1[np.newaxis], y2[np.newaxis]])
    print(y, y_pred, y1, y2)
    x1 = (x1 + 1.) / 2.
    x2 = (x2 + 1.) / 2.
    x = (x + 1.) / 2.
    x_recon1 = (x_recon1 + 1.) / 2.
    x_recon2 = (x_recon2 + 1.) / 2.
    x_recon1 = np.maximum(0., x_recon1)
    x_recon2 = np.maximum(0., x_recon2)
    x_recon1 = np.minimum(1., x_recon1)
    x_recon2 = np.minimum(1., x_recon2)
    a = jaccard_distance(x1, x_recon1).sum()
    b = jaccard_distance(x2, x_recon2).sum()
    print(x1,x2,x,x_recon1,x_recon2)
    fig, ax = plt.subplots(nrows=2, ncols=3, dpi=100, figsize=(40, 10))
    ax[0][0].imshow(x1.squeeze())
    ax[0][1].imshow(x2.squeeze())
    ax[0][2].imshow(x.squeeze())
    ax[1][0].imshow(x_recon1[0].squeeze())
    ax[1][1].imshow(x_recon2[0].squeeze())
    plt.show()
    # dssim = DSSIMObjective()
    # model.compile(optimizer=optimizers.Adam(lr=args.lr),
    #                    loss=[margin_loss, 'mse', dssim],
    #                    loss_weights=[1., args.lam_recon, args.lam_recon],
    #                    metrics={'capsnet': k_categorical_accuracy})
    # e = model.evaluate_generator(buf(data_generator(x_test, y_test, 10, args.overlap), 3), steps=500)
    # print(model.metrics_names, e)


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import fashion_mnist as mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--initial_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--overlap', default=0.8, type=float)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()
    # sample_inputs, sample_outputs = next(data_generator(x_train, y_train, 1, args.overlap))
    # x_sample = sample_inputs[0]
    # y_sample = sample_inputs[1]
    # x_out_sample = sample_outputs[1]

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=(128, 128, 4),
                                                  decoder_output_shape=(128, 128, 4),
                                                  n_class=2,
                                                  routings=args.routings, capsule_size=16)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        #manipulate_latent(manipulate_model, (x_test, y_test), args)
        test_multi(model=eval_model, data=(x_test, y_test), args=args)
