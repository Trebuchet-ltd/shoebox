from typing import List

import pip

try:
    import numpy as np
    from cv2 import cv2
except ImportError:
    pip.main(['install', 'numpy'])
    pip.main(['install', 'opencv-python'])
    pip.main(['install', 'scipy'])


def get_contours_and_crop(images):
    ret_images = []
    bounds = []

    for image in images:
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour = None
        max_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > max_area:
                max_area = w * h
                contour = cnt

        x, y, w, h = cv2.boundingRect(contour)

        ret_images.append(image[y:y + h, x:x + w])
        bounds.append({"x": x, "y": y, "w": w, "h": h})

    return ret_images, bounds


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


def create_cube(images: List[np.ndarray], size: int):
    images, bounds = get_contours_and_crop(images)
    images = align_images(images, size)

    cube = np.tile(images[0], (size, 1, 1)).astype("uint8")
    cube = carve_out(cube, images[1], images[2])

    return cube, bounds
