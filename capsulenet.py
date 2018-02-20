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
from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Mask
from combine_mnist import sample_and_combine
from buffering import buffered_gen_threaded as buf

np.random.seed(0)
K.set_image_data_format('channels_last')

from keras.utils.vis_utils import plot_model


def conv2d_transpose_bn(x, filters, kernel_size, strides=(1, 1), padding='valid', activation='relu', name=''):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation, name=name)(x)
    return x


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

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters_bottleneck, (1, 1), padding='same', name=conv_name_base + '2a', kernel_initializer='he_uniform')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters_bottleneck, kernel_size, kernel_initializer='he_uniform',
               padding='same', name=conv_name_base + '2b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters, (1, 1), padding='same', name=conv_name_base + '2c', kernel_initializer='he_uniform')(x)

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

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)

    shortcut = layers.Conv2DTranspose(filters, (1, 1), strides=strides, padding='same', kernel_initializer='he_uniform',
                      name=conv_name_base + '1')(x)

    x = layers.Conv2DTranspose(filters_bottleneck, (1, 1), strides=strides, padding='same', kernel_initializer='he_uniform',
               name=conv_name_base + '2a')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters_bottleneck, kernel_size, padding='same', kernel_initializer='he_uniform',
               name=conv_name_base + '2b')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2DTranspose(filters, (1, 1), padding='same', name=conv_name_base + '2c', kernel_initializer='he_uniform')(x)

    x = layers.add([x, shortcut])

    return x


def CapsNet(input_shape, n_class, routings, capsule_size=16):
    """
    A Capsule Network on MNIST.
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

    def res_decoder():
        dcaps = layers.Input((n_class, capsule_size), name='masked_digitcaps')
        # pcaps = layers.Input((100352, 8))
        conv = layers.Input((112, 112, 256))

        y = layers.Reshape((1, 1, n_class*capsule_size))(dcaps)

        y = layers.Conv2DTranspose(32, 7)(y)

        y = identity_block(y, 3, 32, '1', '1')
        y = identity_block(y, 3, 32, '1', '2')
        y = conv_block(y, 3, 64, '1', '3')

        y = identity_block(y, 3, 64, '2', '1')
        y = identity_block(y, 3, 64, '2', '2')
        y = conv_block(y, 3, 128, '2', '4')

        y = identity_block(y, 3, 128, '3', '1')
        y = identity_block(y, 3, 128, '3', '2')
        y = conv_block(y, 3, 256, '3', '6')

        y = identity_block(y, 3, 256, '4', '1')
        y = identity_block(y, 3, 256, '4', '2')
        y = conv_block(y, 3, 256, '4', '4')

        y = layers.concatenate([y, conv])

        y = identity_block(y, 3, 512, '5', '1')
        y = identity_block(y, 3, 512, '5', '2')
        y = identity_block(y, 3, 512, '5', '3')

        y = conv_block(y, 3, 128, '5', '4', strides=1)

        y = identity_block(y, 3, 128, '6', '2')
        y = identity_block(y, 3, 128, '6', '3')

        y = conv2d_transpose_bn(y, 4, 17, activation='tanh', name='out_recon')

        m = models.Model([conv, dcaps], y, name='decoder')
        # m.summary()
        plot_model(m, show_shapes=True, to_file='decoder.png')
        return m

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    # out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y1 = layers.Input(shape=(n_class,), name='input_label_1')
    y2 = layers.Input(shape=(n_class,), name='input_label_2')
    masked_by_y1 = Mask()([digitcaps, y1], name='masked_digitcap_1')  # The true label is used to mask the output of capsule layer. For training
    masked_by_y2 = Mask()([digitcaps, y2], name='masked_digitcap_2')  # The true label is used to mask the output of capsule layer. For training
    # masked1 = Mask(1)(digitcaps) # Mask using the capsule with maximal length. For prediction
    # masked2 = Mask(2)(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = res_decoder()

    def make_pose_estimator():
        masked_dcaps = layers.Input((n_class, capsule_size), name='masked_digitcaps')
        pose = layers.Flatten()(masked_dcaps)
        pose = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(pose)
        pose = layers.Dense(1024, activation='relu', kernel_initializer='he_uniform')(pose)
        pose = layers.Dense(3, activation='tanh', name='pose')(pose)
        m = models.Model(masked_dcaps, pose, name='pose')
        plot_model(m, show_shapes=True, to_file='pose-estimator.png')
        return m

    pose_estimator = make_pose_estimator()

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y1, y2], [decoder([conv1, masked_by_y1]), decoder([conv1, masked_by_y2]), pose_estimator(masked_by_y1), pose_estimator(masked_by_y2)])
    eval_model = models.Model([x, y1, y2], [decoder([conv1, masked_by_y1]), decoder([conv1, masked_by_y2]), pose_estimator(masked_by_y1), pose_estimator(masked_by_y2)])
    plot_model(train_model, show_shapes=True)

    return train_model, eval_model


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


def image_loss(y_true, y_pred):
    dssim = DSSIMObjective()
    return dssim(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def preprocess_pose(pose):
    # elevation from -30 to 30 deg
    pose[:, 0] = pose[:, 0] / (np.pi / 6.)
    # azimuth from 0 to 360deg
    pose[:, 1] = pose[:, 1] / np.pi - 1.
    # distance factor from 1.0 to sqrt(2.0)
    half = (np.sqrt(2.) - 1.) / 2.
    pose[:, 2] = (pose[:, 2] - 1. - half) / half
    pose = np.clip(pose, -1., 1.)
    return pose


def data_generator(x_data, y_data, batch_size, overlap, test=False):
    while True:
        data = [sample_and_combine(x_data, y_data, overlap) for i in range(batch_size)]
        # Group
        data = zip(*data)
        # Stack
        data = list(map(np.stack, data))
        x1, x2, x, y1, y2, y, pose1, pose2 = data
        pose1 = preprocess_pose(pose1)
        pose2 = preprocess_pose(pose2)
        inputs = x if test else [x, y1, y2]
        yield inputs, [x1, x2, pose1, pose2]


def train(model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param args: arguments
    :return: The trained model
    """
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='decoder_loss_1',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** (epoch/50)))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[image_loss, image_loss, 'mse', 'mse'],
                  # loss_weights=[0.1, 0.1, 0.1, 1., 1.],
                  )

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=buf(data_generator(None, None, args.batch_size, args.overlap), 3),
                        steps_per_epoch=1000,
                        epochs=args.epochs,
                        initial_epoch=args.initial_epoch,
                        # validation_data=buf(data_generator(x_test, y_test, args.batch_size, args.overlap), 3),
                        # validation_steps=500,
                        callbacks=[log, tb, checkpoint ])
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


def denormalize_image(x):
    x = (x + 1.) / 2.
    x = np.clip(x, 0., 1.)
    return x


def sample(model, args):
    n_samples = 4
    fig, ax = plt.subplots(nrows=n_samples, ncols=3, dpi=100, figsize=(40, 10*n_samples))
    for i in range(n_samples):
        x1, x2, x, y1, y2, y, pose1, pose2 = sample_and_combine(None, None, overlap_factor=args.overlap)
        x_recon1, x_recon2, pose1_p, pose2_p = model.predict_on_batch([x[np.newaxis], y1[np.newaxis], y2[np.newaxis]])
        x1 = denormalize_image(x1)
        x2 = denormalize_image(x2)
        x = denormalize_image(x)
        x_recon1 = denormalize_image(x_recon1)
        x_recon2 = denormalize_image(x_recon2)
        print(x1,x2,x,x_recon1,x_recon2)
        print(preprocess_pose(pose1[np.newaxis]), preprocess_pose(pose2[np.newaxis]), pose1_p, pose2_p)
        #ax[i][0].imshow(x1.squeeze())
        #ax[i][1].imshow(x2.squeeze())
        ax[i][0].imshow(x.squeeze())
        ax[i][1].imshow(x_recon1[0].squeeze())
        ax[i][2].imshow(x_recon2[0].squeeze())
    plt.show()


def get_iou(x_true, x_pred):
    bs = len(x_true)
    alpha_true = x_true[:,:,:,-1].reshape(bs, -1) > 0
    alpha_pred = x_pred[:,:,:,-1].reshape(bs, -1) > 0
    i = alpha_true * alpha_pred
    i = i.sum(axis=-1)
    u = alpha_true + alpha_pred
    u = u.sum(axis=-1)
    iou = i / u
    return iou.mean()


def test_multi(model, args):
    dssim = DSSIMObjective()
    mse1 = 0
    mse2 = 0
    ssim1 = 0
    ssim2 = 0
    iou1 = 0
    iou2 = 0
    error1 = []
    error2 = []

    # Build the computational graph first
    x_true = K.placeholder((None, 128, 128, 4), dtype='float32')
    x_pred = K.placeholder((None, 128, 128, 4), dtype='float32')
    mse = mean_squared_error(x_true, x_pred)
    ssim = dssim(x_true, x_pred)
    sess = K.get_session()

    # x1 background, x2 foreground object
    n = 2500 # @ batch_size 40 = 100k samples
    g = data_generator(None, None, args.batch_size, args.overlap)
    for i in range(n):
        inputs, ground_truth = next(g)
        x1_pred, x2_pred, pose1_pred, pose2_pred = model.predict_on_batch(inputs)
        x1_true, x2_true, pose1_true, pose2_true = ground_truth
        mse1 += sess.run(mse, feed_dict={x_true: x1_true, x_pred: x1_pred}).mean()
        mse2 += sess.run(mse, feed_dict={x_true: x2_true, x_pred: x2_pred}).mean()
        ssim1 += sess.run(ssim, feed_dict={x_true: x1_true, x_pred: x1_pred}).mean()
        ssim2 += sess.run(ssim, feed_dict={x_true: x2_true, x_pred: x2_pred}).mean()
        iou1 += get_iou(x1_true, x1_pred)
        iou2 += get_iou(x2_true, x2_pred)
        e1 = np.absolute(pose1_true - pose1_pred).mean(axis=0) / 2. # percent
        error1.append(e1)
        e2 = np.absolute(pose2_true - pose2_pred).mean(axis=0) / 2. # percent
        error2.append(e2)
        print(i)

    mse1 /= n
    mse2 /= n
    ssim1 /= n
    ssim2 /= n
    iou1 /= n
    iou2 /= n
    error1 = np.stack(error1).mean(axis=0)
    error2 = np.stack(error2).mean(axis=0)
    print(mse1, mse2, ssim1, ssim2, iou1, iou2, error1, error2)
    #print(model.metrics_names, e)


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


if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--initial_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--overlap', default=0.8, type=float)
    parser.add_argument('-s', '--sample', action='store_true',
                        help="Test the trained model on testing dataset, show images")
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

    # define model
    model, eval_model = CapsNet(input_shape=(128, 128, 4), n_class=2, routings=args.routings,
                                                  capsule_size=16)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, args=args)
    elif args.sample:
        sample(model, args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        #manipulate_latent(manipulate_model, (x_test, y_test), args)
        test_multi(model=eval_model, args=args)
