from typing import List

import cv2
import numpy as np


def align_images(images, size):
    h, w = images[0].shape[:2]

    front = np.zeros((size, size))
    side = np.zeros((size, size))
    top = np.zeros((size, size))

    img = images[0]

    front[:img.shape[0], :img.shape[1]] = img

    dh, dw = images[1].shape[:2]
    img = cv2.resize(images[1], (int((dw / dh) * h) or 1, h or 1))
    side[:img.shape[0], :img.shape[1]] = img

    img = cv2.resize(images[2], (w or 1, int((dw / dh) * h) or 1))
    top[:img.shape[0], :img.shape[1]] = img

    return [front, side, top]


def carve_out(cube, img_side, img_top):
    for row in range(img_side.shape[0]):
        for col in range(img_side.shape[1]):
            if img_side[row, col] == 0:
                cube[col, row, :] = 0
            if img_top[row, col] == 0:
                cube[row, :, col] = 0

    return cube


def create_cube(images: List[np.ndarray], size: int) -> np.ndarray:
    images = align_images(images, size)

    cube = np.tile(images[0], (size, 1, 1))
    cube = carve_out(cube, images[1], images[2])

    return cube
