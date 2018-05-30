"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python test.py
       python test.py --epochs 50
       python test.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np

from keras import backend as K
from keras.losses import mean_squared_error
from keras_contrib.losses import DSSIMObjective

import matplotlib.pyplot as plt

from vignet.data import data_generator
from vignet.models import VIGNet

K.set_image_data_format('channels_last')

def denormalize_image(x):
    x = (x + 1.) / 2.
    x = np.clip(x, 0., 1.)
    return x


def sample(model, args):
    n_samples = 4
    fig, ax = plt.subplots(nrows=n_samples, ncols=3, dpi=100, figsize=(40, 10*n_samples))
    g = data_generator(1, test=True)
    for i in range(n_samples):
        inputs, ground_truth = next(g)
        x = inputs[0]
        x1_pred, x2_pred, pose1_pred, pose2_pred = model.predict_on_batch(inputs)
        x1_true, x2_true, pose1_true, pose2_true = ground_truth
        # Denormalize all images
        x, x1_true, x2_true, x1_pred, x2_pred = map(denormalize_image, [x, x1_true, x2_true, x1_pred, x2_pred])
        print(pose1_true, pose1_pred, pose2_true, pose2_pred)
        ax[i][0].imshow(x.squeeze())
        ax[i][1].imshow(x1_pred[0].squeeze())
        ax[i][2].imshow(x2_pred[0].squeeze())
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
    g = data_generator(args.batch_size, test=True)
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


def main():
    import os
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="VIGNet on ShapeNet.")
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--initial_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    train_model, test_model, voxelizer = VIGNet(input_shape=(128, 128, 4), n_class=2, routings=args.routings, capsule_size=16)
    test_model.summary()

    train_model.load_weights('pretrained/weights-905.h5')
    voxelizer.load_weights('pretrained/voxelizer-weights.h5')


    test_multi(model=train_model, args=args)


if __name__ == '__main__':
    main()
