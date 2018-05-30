#!/usr/bin/env python3

import os.path
import numpy as np

from vignet.combine import sample_and_combine


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


def preprocess_img(x):
    x = x / 127.5 - 1.
    return x


def load_archive(prefix, name, test):
    suffix = 'test' if test else 'train'
    path = os.path.join(prefix, name + '.' + suffix + '.npz')
    print(path)
    ar = np.load(path)
    img = ar['images']
    pose = ar['labels']
    return img, pose


def data_generator(batch_size, seed=0, test=False, preprocess=True):
    prefix = '/mnt/data/Projects/datasets/shapenet/3d-r2n2-dataset/'
    car_data = load_archive(prefix, 'car', test)
    motorcycle_data = load_archive(prefix, 'motorcycle', test)
    rng = np.random.RandomState(seed)
    while True:
        data = [sample_and_combine(car_data, motorcycle_data, rng) for i in range(batch_size)]
        # Group
        data = zip(*data)
        # Stack
        data = list(map(np.stack, data))
        x1, x2, x, y1, y2, y, pose1, pose2 = data
        # Preprocess
        if preprocess:
            x1 = preprocess_img(x1)
            x2 = preprocess_img(x2)
            x = preprocess_img(x)
            pose1 = preprocess_pose(pose1)
            pose2 = preprocess_pose(pose2)
        yield [x, y1, y2], [x1, x2, pose1, pose2]
