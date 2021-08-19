import time

import math
import threading
import numpy as np
import cv2
from scipy import signal

try:
    import bpy
except ImportError:
    pass


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

    # Orange color
    # R = 255
    # G = 153
    # B = 20
    # img = np.where(
    #     (
    #         ((img[..., [0]] > B-1) & (img[..., [0]] < B+1)) &
    #         ((img[..., [1]] > G-1) & (img[..., [1]] < G+1)) &
    #         ((img[..., [2]] > R-1) & (img[..., [2]] < R+1))
    #     ),
    #     (255, 255, 255), img)

    # CMY done

    imgarr = np.where(
        (
                ((img[..., [2]] - img[..., [1]] < thresh) | (img[..., [2]] - img[..., [0]] < thresh)) &
                ((img[..., [0]] - img[..., [1]] < thresh) | (img[..., [1]] - img[..., [0]] < thresh)) &
                ((img[..., [0]] - img[..., [2]] < thresh) | (img[..., [2]] - img[..., [0]] < thresh))
        ),
        (255, 255, 255), img)
    return imgarr


def skin_color_detection(img, type=0, sp=40):
    if type == 0:
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
    sizes = []

    thresholds = [29, 19, 28]

    images[2] = rotate_image(images[2],-45)
    index = 0
    for image in images:

        image = skin_color_detection(check_and_change_frame(image), 0, thresholds[index])
        index += 1

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour = None
        max_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > max_area:
                max_area = w * h
                contour = cnt

        x, y, w, h = cv2.boundingRect(contour)

        output.append(thresh[y:y + h, x:x + w])
        sizes.append((w, h))

    return output, sizes


def align_images(images):
    h, w = images[0].shape[:2]

    front = np.zeros(SIZE)
    side = np.zeros(SIZE)
    top = np.zeros(SIZE)

    img = images[0]
    front[:img.shape[0], :img.shape[1]] = img

    dh, dw = images[1].shape[:2]
    img = cv2.resize(images[1], (int((dw / dh) * h), h))
    print(h, w, dh, dw, img.shape, int((dw / dh) * h))
    side[:img.shape[0], :img.shape[1]] = img

    img = cv2.resize(images[2], (w, int((dw / dh) * h)))
    top[:img.shape[0], :img.shape[1]] = img

    return [front, side, top]


def read_images(videos):
    images = []

    for video in videos:
        ret, img = video.read()
        if not ret:
            return []

        images.append(cv2.resize(img, SIZE))

    return images


def carve_out(cube, img_side, img_top):
    for row in range(img_side.shape[0]):
        for col in range(img_side.shape[1]):
            if img_side[row, col] == 0:
                cube[col, row, :] = 0
            if img_top[row, col] == 0:
                cube[row, :, col] = 0

    return cube


def draw_mesh(vertex_dict, vertex_array):
    faces = []

    mesh = bpy.data.meshes.new("myBeautifulMesh")  # add the new mesh
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    start_time = time.monotonic()

    adders = {
        # x
        (0, 0): [0, 1, 0],
        (0, 1): [0, 0, 1],
        (0, 2): [0, 1, 1],
        # y
        (1, 0): [1, 0, 0],
        (1, 1): [0, 0, 1],
        (1, 2): [1, 0, 1],
        # z
        (2, 0): [0, 1, 0],
        (2, 1): [1, 0, 0],
        (2, 2): [1, 1, 0],
    }

    for key in vertex_dict:
        for i in range(3):
            square = [vertex_dict[key]]
            for j in range(3):
                adder = adders[(i, j)]
                vert = vertex_dict.get((key[0] + adder[0], key[1] + adder[1], key[2] + adder[2]))

                if vert:
                    square.append(vert)
            if len(square) == 4:
                faces.append((square[0], square[1], square[3], square[2]))

    print("Skinning :", time.monotonic() - start_time)

    mesh.from_pydata(vertex_array, [], faces)

    bpy.context.view_layer.objects.active = obj


def get_edges(cube):
    actives_dict = {}
    actives_array = []

    n = 2
    kern = np.full((n + 1, n + 1, n + 1), -1)
    kern[1, 1, 1] = 26

    # 3d convolve
    cube = signal.convolve(cube, kern, mode='same')

    for layer in range(cube.shape[0]):
        for row in range(cube.shape[1]):
            for col in range(cube.shape[2]):
                if cube[layer][row][col] > 0.01:
                    actives_array.append((layer, row, col))
                    actives_dict[(layer, row, col)] = len(actives_array) - 1

    return actives_dict, actives_array


def calculate_angles(current_sizes, previous_sizes):
    return [previous_sizes[i][0] / current_sizes[i][0] for i in range(len(current_sizes))]


def run_in_loop(prev_sizes, videos):
    images = read_images(videos)

    if len(images) == 0:
        return False

    images, sizes = remove_background(images)

    angles = calculate_angles(sizes, prev_sizes)

    print(angles)

    return True, sizes


def main():
    videos = [cv2.VideoCapture("data/front.mp4"), cv2.VideoCapture("data/side.mp4"), cv2.VideoCapture("data/top.mp4")]

    for i in range(10):
        read_images(videos)

    start_time = time.monotonic()
    images = read_images(videos)
    images, sizes = remove_background(images)
    print("Remove Bg :", time.monotonic() - start_time)

    start_time = time.monotonic()
    images = align_images(images)
    print("Align :", time.monotonic() - start_time)

    cube = np.tile(images[0], (SIZE[0], 1, 1))

    start_time = time.monotonic()
    cube = carve_out(cube, images[1], images[2])
    print("Carving out :", time.monotonic() - start_time)

    start_time = time.monotonic()
    vertex_dict, vertex_array = get_edges(cube)
    print("Removing inner points :", time.monotonic() - start_time)

    start_time = time.monotonic()
    draw_mesh(vertex_dict, vertex_array)
    print("Blender + skinning:", time.monotonic() - start_time)

    run = True
    while run:
        run, sizes = run_in_loop(sizes, videos)

    for video in videos:
        video.release()


if __name__ == '__main__':
    SIZE = (64, 64)

    t1 = threading.Thread(target=main)
    t1.start()
