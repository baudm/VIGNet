#!/usr/bin/env python3

from keras import callbacks
from keras.optimizers import Adam

from vignet.data import data_generator
from vignet.losses import image_loss
from vignet.models import VIGNet


def train(model, args):
    """
    Training a VIGNet
    :param model: the VIGNet model
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
    model.compile(optimizer=Adam(lr=args.lr),
                  loss=[image_loss, image_loss, 'mse', 'mse'],
                  # loss_weights=[0.1, 0.1, 0.1, 1., 1.],
                  )

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=data_generator(args.batch_size),
                        steps_per_epoch=1000,
                        epochs=args.epochs,
                        initial_epoch=args.initial_epoch,
                        # validation_data=buf(data_generator(args.batch_size), 3),
                        # validation_steps=500,
                        callbacks=[log, tb, checkpoint ])
    # End: Training with data augmentation -----------------------------------------------------------------------#

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    #from utils import plot_log
    #plot_log(args.save_dir + '/log.csv', show=True)

    return model


def main():
    import os
    import argparse

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
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
    model = VIGNet(input_shape=(128, 128, 4), n_class=2, routings=args.routings, capsule_size=16)[0]
    model.summary()

    # train
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)

    train(model=model, args=args)


if __name__ == '__main__':
    main()
