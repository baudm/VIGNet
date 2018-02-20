#!/usr/bin/env python

import numpy as np


def sample_and_combine(car_data, motorcycle_data, rng):
    car_img, car_pose = car_data
    motor_img, motor_pose = motorcycle_data

    N = len(car_img)

    # Sample car
    n = rng.randint(N)
    x1 = car_img[n]
    pose1 = car_pose[n]
    y1 = np.array([1, 0])

    # Sample motorcycle
    n = rng.randint(N)
    x2 = motor_img[n]
    pose2 = motor_pose[n]
    y2 = np.array([0, 1])

    # Swap 50% of the time
    data = [(x1, pose1, y1), (x2, pose2, y2)]
    rng.shuffle(data)
    (x1, pose1, y1), (x2, pose2, y2) = data

    y_pad = [(0, 128 - 96), (128 - 96, 0)]
    rng.shuffle(y_pad)
    y_pad1, y_pad2 = y_pad

    x1 = np.pad(x1, (y_pad1, (0, 0), (0, 0)), mode='constant', constant_values=0)
    x2 = np.pad(x2, (y_pad2, (0, 0), (0, 0)), mode='constant', constant_values=0)
    y = y1.copy()
    y[np.argmax(y2)] = 1

    # Draw x2 on top of x1
    combined = x1.copy()
    mask = x2[:, :, -1] > 0
    mask = np.dstack([mask, mask, mask, mask])
    combined[mask] = x2[mask]

    return x1, x2, combined, y1, y2, y, pose1, pose2


# def main():
#     bb_w = [0 for i in range(10)]
#     bb_h = [0 for i in range(10)]
#     (x, y), (_, _) = mnist.load_data()
#     for i in range(len(x)):
#         x1 = x[i]
#         y1 = y[i]
#         left_x = ((x1 > 0).argmax(axis=0) > 0).argmax()
#         top_y = ((x1 > 0).argmax(axis=1) > 0).argmax()
#         x1 = np.fliplr(x1)
#         x1 = np.flipud(x1)
#         right_x = 28 - ((x1 > 0).argmax(axis=0) > 0).argmax()
#         bottom_y = 28 - ((x1 > 0).argmax(axis=1) > 0).argmax()
#         w = right_x - left_x
#         h = bottom_y - top_y
#         bb_w[y1] += w
#         bb_h[y1] += h
#
#     for i in range(10):
#         bb_w[i] /= 6000.
#         bb_h[i] /= 6000.
#
#     return bb_w, bb_h

import glob
from skimage.io import imread


def main2():
    bb_w = 0
    bb_h = 0
    count = 0

    for img in glob.iglob('/home/darwin/Projects/datasets/shapenet/transparent/**/*.png', recursive=True):
        x1 = imread(img)
        try:
            height, width, d = x1.shape
        except ValueError:
            print(img)
        x1 = np.split(x1, d, axis=-1)[-1].squeeze()
        left_x = ((x1 > 0).argmax(axis=0) > 0).argmax()
        top_y = ((x1 > 0).argmax(axis=1) > 0).argmax()
        x1 = np.fliplr(x1)
        x1 = np.flipud(x1)
        right_x = width - ((x1 > 0).argmax(axis=0) > 0).argmax()
        bottom_y = height - ((x1 > 0).argmax(axis=1) > 0).argmax()
        w = right_x - left_x
        h = bottom_y - top_y
        bb_w += w
        bb_h += h
        count += 1

    bb_h /= count
    bb_w /= count

    print(bb_w, bb_h)



if __name__ == '__main__':
    main2()
