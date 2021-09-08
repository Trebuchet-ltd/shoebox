import threading
from typing import List, Tuple

import numpy as np
import cv2
from scipy import signal

try:
    import bpy
    import mathutils
    from cube import create_cube
    from image import get_next_processed_frame
    from bone import create_armature
    from loop import run_in_loop
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


def set_material(mesh):
    mat = bpy.data.materials.get("Material")

    if mesh.data.materials:
        # assign to 1st material slot
        mesh.data.materials[0] = mat
    else:
        # no slots
        mesh.data.materials.append(mat)


def scale_shoe(mesh, points):
    bpy.ops.object.mode_set(mode='OBJECT')

    shoe = bpy.data.objects["shoe"]
    shoe.scale *= ((mesh.dimensions[0] / shoe.dimensions[0]) * 1.25)
    shoe.location += (mathutils.Vector(points[1][0]) + mathutils.Vector(points[1][1])) / 2

    return shoe


def set_parent(child, parent):
    for obj in bpy.data.objects:
        obj.select_set(False)

    child.select_set(True)
    bpy.data.objects[parent].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[parent]

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.parent_set(type='ARMATURE_AUTO', keep_transform=False)


def parent_mesh(mesh, shoe, armature):
    bpy.data.objects[armature].data.edit_bones['bone1'].parent = \
        bpy.data.objects[armature].data.edit_bones['bone0']

    set_parent(shoe, armature)
    set_parent(mesh, armature)

    bpy.data.objects[armature].select_set(True)


def get_bone_rotation(bone):
    mat = bone.matrix.to_euler()
    return [math.degrees(mat.x), math.degrees(mat.y), math.degrees(mat.z)]


def worker(image_generator, bounds, size):
    initial_angles = get_bone_rotation(bpy.data.objects["Armature"].pose.bones[1])
    for images in image_generator:
        bounds = run_in_loop(images, size, bpy.data.objects["Armature"].pose.bones, initial_angles, bounds)


def main(video_paths, size=64):
    videos = [cv2.VideoCapture(path) for path in video_paths]
    image_generator = get_next_processed_frame(videos, (size, size))

    for i in range(10):
        next(image_generator)

    cube, bounds = create_cube(next(image_generator)[0], size)
    vertex_dict, vertex_array = get_edges(cube)

    armature, points = create_armature(cube, size)
    mesh = draw_mesh(vertex_dict, vertex_array)
    set_material(mesh)

    shoe = scale_shoe(mesh, points)
    parent_mesh(mesh, shoe, armature)

    bpy.ops.object.mode_set(mode='POSE')
    t1 = threading.Thread(target=worker, args=(image_generator, bounds, size))
    t1.start()


if __name__ == '__main__':
    main(["data/front.mp4", "data/side.mp4", "data/top.mp4"])
