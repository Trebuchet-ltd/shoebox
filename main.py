import threading
from typing import List, Tuple

import numpy as np
import cv2
from scipy import signal

try:
    import bpy
    from cube import create_cube
    from image import get_next_processed_frame
except ImportError:
    pass


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


def draw_mesh(vertex_dict, vertex_array):
    faces = []

    mesh = bpy.data.meshes.new("myBeautifulMesh")  # add the new mesh
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

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

    mesh.from_pydata(vertex_array, [], faces)

    return obj


def create_bones(points: List[List[Tuple[float, float, float]]]):
    bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    arm_obj = bpy.data.objects["Armature"].data

    # must be in edit mode to add bones
    index = 0

    for point in points:
        b = arm_obj.edit_bones.new(f'bone{index}')
        # a new bone will have zero length and not be kept
        # move the head/tail to keep the bone
        b.head = point[0]
        b.tail = point[1]

        print(point)

        index += 1

    for obj in arm_obj.edit_bones:
        obj.select_head = False
        obj.select_tail = False

    return arm_obj


def parent_mesh(mesh, armature):
    for obj in bpy.data.objects:
        obj.select_set(False)

    bpy.data.objects["Armature"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects["Armature"].select_set(True)
    # the active object will be the parent of all selected object
    bpy.context.view_layer.objects.active = bpy.data.objects["Armature"]

    bpy.ops.object.parent_set(type='BONE', keep_transform=True)


def calculate_angles(images):
    angles = []

    for image in images:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        segmented = max(contours, key=cv2.contourArea)
        _, _, angle = cv2.minAreaRect(segmented)

        angles.append(angle)

    return angles


def collapse_cube(cube):
    p_sum = 0

    layer = cube.shape[1] - 1

    for i in range(cube.shape[1]):
        c_sum = np.sum(np.sum(cube[:, i, :]))

        if p_sum != 0 and abs(c_sum - p_sum) > 35:
            layer = i
            break

        p_sum = c_sum

    return [np.sum(cube[:, layer:, :], axis=i, dtype="uint8") for i in range(3)]


def run_in_loop(images, size):
    cube = create_cube(images, size)

    shadows = collapse_cube(cube)
    angles = calculate_angles(shadows)

    cv2.waitKey()


def get_nonzero_bounds(image):
    left, right = -1, -1
    top, bottom = -1, -1

    for i in range(image.shape[0]):
        if any(image[i]):
            if top == -1:
                top = i
            else:
                bottom = i

    for i in range(image.shape[1]):
        if any(image[i]):
            if left == -1:
                left = i
            else:
                right = i

    return {"top": top, "bottom": bottom, "left": left, "right": right}


def get_mid_point(points):
    x = (points["left"] + points["right"]) / 2
    y = (points["top"] + points["bottom"]) / 2

    return {"x": x, "y": y}


def main(size=64):
    videos = [cv2.VideoCapture(path) for path in ["data/front.mp4", "data/side.mp4", "data/top.mp4"]]
    image_generator = get_next_processed_frame(videos, (size, size))

    cube = create_cube(next(image_generator), size)

    bounds = [get_nonzero_bounds(image) for image in [cube[10, :, :], cube[:, 10, :], cube[:, :, 10]]]
    mid_points = [get_mid_point(bound) for bound in bounds]

    armature = create_bones([
        [
            (bounds[1]["bottom"], bounds[2]["left"], mid_points[0]["y"] - size / 2),
            (bounds[1]["bottom"], bounds[1]["right"], mid_points[0]["y"] - size / 2)
        ],
        [
            (bounds[1]["bottom"], bounds[1]["right"], mid_points[0]["y"] - size / 2),
            (bounds[1]["top"], bounds[2]["right"], mid_points[0]["y"] - size / 2)
        ]
    ])

    vertex_dict, vertex_array = get_edges(cube)
    mesh = draw_mesh(vertex_dict, vertex_array)

    parent_mesh(mesh, armature)

    for images in image_generator:
        run_in_loop(images, size)

    for video in videos:
        video.release()


if __name__ == '__main__':
    t1 = threading.Thread(target=main)
    t1.start()