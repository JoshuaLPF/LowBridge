from __future__ import annotations

import numpy as np
from scipy import ndimage


def random_rot_flip(image, mask, k=None, axis=None):
    if k is None:
        k = np.random.randint(0, 4)
    image, mask = np.rot90(image, k), np.rot90(mask, k)
    if axis is None:
        axis = np.random.randint(0, 2)
    return (np.flip(image, axis=axis).copy(),
            np.flip(mask, axis=axis).copy(), k, axis)


def random_rotate(image, mask, angle=None):
    if angle is None:
        angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=1, reshape=False)
    mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    return image, mask, angle
