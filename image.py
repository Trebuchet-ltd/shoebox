import cv2
import numpy as np
from typing import List

from cv2 import VideoCapture


def read_images(videos, size):
    images = []

    for video in videos:
        ret, img = video.read()
        if not ret:
            return None
        images.append(cv2.resize(img, size))

    return images


def check_and_change_frame(frame, thresh=7):
    img = frame
    thresh1 = 40

    img = np.where(
        (
            ((img[..., [1]] - img[..., [0]] > thresh1) & (img[..., [1]] - img[..., [2]] > thresh1))

        ),
        (255, 255, 255), img)

    img = np.where(
        (
            ((img[..., [0]] - img[..., [1]] > thresh1) & (img[..., [0]] - img[..., [2]] > thresh1))

        ),
        (255, 255, 255), img)

    # RGB DONE
    thresh2 = 40
    img = np.where(
        (
            ((img[..., [0]] - img[..., [2]] > thresh2) & (img[..., [1]] - img[..., [2]] > thresh2))

        ),
        (255, 255, 255), img)
    img = np.where(
        (
            ((img[..., [2]] - img[..., [1]] > thresh2) & (img[..., [0]] - img[..., [1]] > thresh2))

        ),
        (255, 255, 255), img)
    img = np.where(
        (
            ((img[..., [1]] - img[..., [0]] > thresh2) & (img[..., [2]] - img[..., [0]] > thresh2))

        ),
        (255, 255, 255), img)

    img_array = np.where(
        (
                ((img[..., [2]] - img[..., [1]] < thresh) | (img[..., [2]] - img[..., [0]] < thresh)) &
                ((img[..., [0]] - img[..., [1]] < thresh) | (img[..., [1]] - img[..., [0]] < thresh)) &
                ((img[..., [0]] - img[..., [2]] < thresh) | (img[..., [2]] - img[..., [0]] < thresh))
        ),
        (255, 255, 255), img)
    return img_array


def skin_color_detection(img, replace_type=0, sp=40):
    if replace_type == 0:
        replace1 = (0, 0, 0)
        replace2 = img
    else:
        replace1 = (255, 255, 255)
        replace2 = (0, 0, 0)
        sp += 30
    img = np.where(
        (
            (

                    (
                            ((img[..., [0]] < (153 - sp)) | (img[..., [0]] > (153 + sp))) |
                            ((img[..., [1]] < (193 - sp)) | (img[..., [1]] > (193 + sp))) |
                            ((img[..., [2]] < (255 - sp)) | (img[..., [2]] > (255 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (178 - sp)) | (img[..., [0]] > (178 + sp))) |
                            ((img[..., [1]] < (209 - sp)) | (img[..., [1]] > (209 + sp))) |
                            ((img[..., [2]] < (255 - sp)) | (img[..., [2]] > (255 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (146 - sp)) | (img[..., [0]] > (146 + sp))) |
                            ((img[..., [1]] < (185 - sp)) | (img[..., [1]] > (185 + sp))) |
                            ((img[..., [2]] < (243 - sp)) | (img[..., [2]] > (243 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (172 - sp)) | (img[..., [0]] > (172 + sp))) |
                            ((img[..., [1]] < (202 - sp)) | (img[..., [1]] > (202 + sp))) |
                            ((img[..., [2]] < (246 - sp)) | (img[..., [2]] > (246 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (166 - sp)) | (img[..., [0]] > (166 + sp))) |
                            ((img[..., [1]] < (195 - sp)) | (img[..., [1]] > (195 + sp))) |
                            ((img[..., [2]] < (238 - sp)) | (img[..., [2]] > (238 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (139 - sp)) | (img[..., [0]] > (139 + sp))) |
                            ((img[..., [1]] < (176 - sp)) | (img[..., [1]] > (176 + sp))) |
                            ((img[..., [2]] < (232 - sp)) | (img[..., [2]] > (232 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (113 - sp)) | (img[..., [0]] > (113 + sp))) |
                            ((img[..., [1]] < (158 - sp)) | (img[..., [1]] > (158 + sp))) |
                            ((img[..., [2]] < (226 - sp)) | (img[..., [2]] > (226 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (88 - sp)) | (img[..., [0]] > (88 + sp))) |
                            ((img[..., [1]] < (141 - sp)) | (img[..., [1]] > (141 + sp))) |
                            ((img[..., [2]] < (221 - sp)) | (img[..., [2]] > (221 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (160 - sp)) | (img[..., [0]] > (160 + sp))) |
                            ((img[..., [1]] < (188 - sp)) | (img[..., [1]] > (188 + sp))) |
                            ((img[..., [2]] < (229 - sp)) | (img[..., [2]] > (229 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (132 - sp)) | (img[..., [0]] > (132 + sp))) |
                            ((img[..., [1]] < (167 - sp)) | (img[..., [1]] > (167 + sp))) |
                            ((img[..., [2]] < (221 - sp)) | (img[..., [2]] > (221 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (106 - sp)) | (img[..., [0]] > (106 + sp))) |
                            ((img[..., [1]] < (148 - sp)) | (img[..., [1]] > (148 + sp))) |
                            ((img[..., [2]] < (212 - sp)) | (img[..., [2]] > (212 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (81 - sp)) | (img[..., [0]] > (81 + sp))) |
                            ((img[..., [1]] < (130 - sp)) | (img[..., [1]] > (130 + sp))) |
                            ((img[..., [2]] < (204 - sp)) | (img[..., [2]] > (204 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (74 - sp)) | (img[..., [0]] > (74 + sp))) |
                            ((img[..., [1]] < (119 - sp)) | (img[..., [1]] > (119 + sp))) |
                            ((img[..., [2]] < (187 - sp)) | (img[..., [2]] > (187 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (99 - sp)) | (img[..., [0]] > (99 + sp))) |
                            ((img[..., [1]] < (138 - sp)) | (img[..., [1]] > (138 + sp))) |
                            ((img[..., [2]] < (198 - sp)) | (img[..., [2]] > (198 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (125 - sp)) | (img[..., [0]] > (125 + sp))) |
                            ((img[..., [1]] < (159 - sp)) | (img[..., [1]] > (159 + sp))) |
                            ((img[..., [2]] < (209 - sp)) | (img[..., [2]] > (209 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (154 - sp)) | (img[..., [0]] > (154 + sp))) |
                            ((img[..., [1]] < (181 - sp)) | (img[..., [1]] > (181 + sp))) |
                            ((img[..., [2]] < (221 - sp)) | (img[..., [2]] > (221 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (119 - sp)) | (img[..., [0]] > (119 + sp))) |
                            ((img[..., [1]] < (150 - sp)) | (img[..., [1]] > (150 + sp))) |
                            ((img[..., [2]] < (198 - sp)) | (img[..., [2]] > (198 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (92 - sp)) | (img[..., [0]] > (92 + sp))) |
                            ((img[..., [1]] < (128 - sp)) | (img[..., [1]] > (128 + sp))) |
                            ((img[..., [2]] < (184 - sp)) | (img[..., [2]] > (184 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (68 - sp)) | (img[..., [0]] > (68 + sp))) |
                            ((img[..., [1]] < (108 - sp)) | (img[..., [1]] > (108 + sp))) |
                            ((img[..., [2]] < (170 - sp)) | (img[..., [2]] > (170 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (112 - sp)) | (img[..., [0]] > (112 + sp))) |
                            ((img[..., [1]] < (142 - sp)) | (img[..., [1]] > (142 + sp))) |
                            ((img[..., [2]] < (187 - sp)) | (img[..., [2]] > (187 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (85 - sp)) | (img[..., [0]] > (85 + sp))) |
                            ((img[..., [1]] < (119 - sp)) | (img[..., [1]] > (119 + sp))) |
                            ((img[..., [2]] < (170 - sp)) | (img[..., [2]] > (170 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (61 - sp)) | (img[..., [0]] > (61 + sp))) |
                            ((img[..., [1]] < (97 - sp)) | (img[..., [1]] > (97 + sp))) |
                            ((img[..., [2]] < (153 - sp)) | (img[..., [2]] > (153 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (40 - sp)) | (img[..., [0]] > (40 + sp))) |
                            ((img[..., [1]] < (78 - sp)) | (img[..., [1]] > (78 + sp))) |
                            ((img[..., [2]] < (136 - sp)) | (img[..., [2]] > (136 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (23 - sp)) | (img[..., [0]] > (23 + sp))) |
                            ((img[..., [1]] < (61 - sp)) | (img[..., [1]] > (61 + sp))) |
                            ((img[..., [2]] < (119 - sp)) | (img[..., [2]] > (119 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (10 - sp)) | (img[..., [0]] > (10 + sp))) |
                            ((img[..., [1]] < (46 - sp)) | (img[..., [1]] > (46 + sp))) |
                            ((img[..., [2]] < (102 - sp)) | (img[..., [2]] > (102 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (0 - sp)) | (img[..., [0]] > (0 + sp))) |
                            ((img[..., [1]] < (34 - sp)) | (img[..., [1]] > (34 + sp))) |
                            ((img[..., [2]] < (85 - sp)) | (img[..., [2]] > (85 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (0 - sp)) | (img[..., [0]] > (0 + sp))) |
                            ((img[..., [1]] < (22 - sp)) | (img[..., [1]] > (22 + sp))) |
                            ((img[..., [2]] < (56 - sp)) | (img[..., [2]] > (56 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (7 - sp)) | (img[..., [0]] > (7 + sp))) |
                            ((img[..., [1]] < (35 - sp)) | (img[..., [1]] > (35 + sp))) |
                            ((img[..., [2]] < (76 - sp)) | (img[..., [2]] > (76 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (19 - sp)) | (img[..., [0]] > (19 + sp))) |
                            ((img[..., [1]] < (50 - sp)) | (img[..., [1]] > (50 + sp))) |
                            ((img[..., [2]] < (96 - sp)) | (img[..., [2]] > (96 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (34 - sp)) | (img[..., [0]] > (34 + sp))) |
                            ((img[..., [1]] < (67 - sp)) | (img[..., [1]] > (67 + sp))) |
                            ((img[..., [2]] < (116 - sp)) | (img[..., [2]] > (116 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (54 - sp)) | (img[..., [0]] > (54 + sp))) |
                            ((img[..., [1]] < (87 - sp)) | (img[..., [1]] > (87 + sp))) |
                            ((img[..., [2]] < (136 - sp)) | (img[..., [2]] > (136 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (77 - sp)) | (img[..., [0]] > (77 + sp))) |
                            ((img[..., [1]] < (109 - sp)) | (img[..., [1]] > (109 + sp))) |
                            ((img[..., [2]] < (155 - sp)) | (img[..., [2]] > (155 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (105 - sp)) | (img[..., [0]] > (105 + sp))) |
                            ((img[..., [1]] < (133 - sp)) | (img[..., [1]] > (133 + sp))) |
                            ((img[..., [2]] < (175 - sp)) | (img[..., [2]] > (175 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (14 - sp)) | (img[..., [0]] > (14 + sp))) |
                            ((img[..., [1]] < (38 - sp)) | (img[..., [1]] > (38 + sp))) |
                            ((img[..., [2]] < (73 - sp)) | (img[..., [2]] > (73 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (28 - sp)) | (img[..., [0]] > (28 + sp))) |
                            ((img[..., [1]] < (55 - sp)) | (img[..., [1]] > (55 + sp))) |
                            ((img[..., [2]] < (96 - sp)) | (img[..., [2]] > (96 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (47 - sp)) | (img[..., [0]] > (47 + sp))) |
                            ((img[..., [1]] < (76 - sp)) | (img[..., [1]] > (76 + sp))) |
                            ((img[..., [2]] < (119 - sp)) | (img[..., [2]] > (119 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (70 - sp)) | (img[..., [0]] > (70 + sp))) |
                            ((img[..., [1]] < (99 - sp)) | (img[..., [1]] > (99 + sp))) |
                            ((img[..., [2]] < (141 - sp)) | (img[..., [2]] > (141 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (98 - sp)) | (img[..., [0]] > (98 + sp))) |
                            ((img[..., [1]] < (124 - sp)) | (img[..., [1]] > (124 + sp))) |
                            ((img[..., [2]] < (164 - sp)) | (img[..., [2]] > (164 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (10 - sp)) | (img[..., [0]] > (10 + sp))) |
                            ((img[..., [1]] < (26 - sp)) | (img[..., [1]] > (26 + sp))) |
                            ((img[..., [2]] < (51 - sp)) | (img[..., [2]] > (51 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (22 - sp)) | (img[..., [0]] > (22 + sp))) |
                            ((img[..., [1]] < (44 - sp)) | (img[..., [1]] > (44 + sp))) |
                            ((img[..., [2]] < (76 - sp)) | (img[..., [2]] > (76 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (40 - sp)) | (img[..., [0]] > (40 + sp))) |
                            ((img[..., [1]] < (65 - sp)) | (img[..., [1]] > (65 + sp))) |
                            ((img[..., [2]] < (102 - sp)) | (img[..., [2]] > (102 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (63 - sp)) | (img[..., [0]] > (63 + sp))) |
                            ((img[..., [1]] < (89 - sp)) | (img[..., [1]] > (89 + sp))) |
                            ((img[..., [2]] < (127 - sp)) | (img[..., [2]] > (127 + sp)))
                    ) &
                    (
                            ((img[..., [0]] < (91 - sp)) | (img[..., [0]] > (91 + sp))) |
                            ((img[..., [1]] < (116 - sp)) | (img[..., [1]] > (116 + sp))) |
                            ((img[..., [2]] < (153 - sp)) | (img[..., [2]] > (153 + sp)))
                    )
            )
        ),
        replace1, replace2)
    img = img.astype(np.uint8)
    return img


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def remove_background(images):
    output = []

    thresholds = [29, 19, 28]

    images[2] = rotate_image(images[2], -45)
    index = 0
    for image in images:

        image = skin_color_detection(check_and_change_frame(image), 0, thresholds[index])
        index += 1

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)

        output.append(thresh)

    return output


def get_next_processed_frame(videos: List[VideoCapture], size: (int, int)):
    while all([video.isOpened() for video in videos]):
        images = read_images(videos, size)

        if images is None:
            break

        yield remove_background(images)
