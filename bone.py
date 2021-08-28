import numpy as np
from typing import List,Tuple


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

    ret = {"top": top, "bottom": bottom, "left": left, "right": right}

    for key in ret:
        if ret[key] == -1:
            ret[key] = None

    return ret


def get_mid_point(points):
    x = (points["left"] + points["right"]) / 2
    y = (points["top"] + points["bottom"]) / 2

    return {"x": x, "y": y}


def get_bone_points(cube, size):
    front = get_nonzero_bounds(cube[0, :, :])
    top = get_nonzero_bounds(cube[:, 20, :])
    side = {"top": size, "bottom": 0, "left": size, "right": 0}

    for i in range(size):
        bounds = get_nonzero_bounds(cube[:, :, i])
        side["top"] = min(side["top"], bounds["top"] or size)
        side["right"] = max(side["right"], bounds["right"] or -1)
        side["bottom"] = max(side["bottom"], bounds["bottom"] or -1)
        side["left"] = min(side["left"], bounds["left"] or size)

    mid_points = [get_mid_point(bound) for bound in [front, side, top]]

    return [
        [
            (side["right"], side["top"], mid_points[2]["y"] - size / 4),
            (side["right"], side["bottom"], mid_points[2]["y"] - size / 4)
        ],
        [
            (side["right"], side["bottom"], mid_points[2]["y"] - size / 4),
            (side["left"], mid_points[0]["y"], mid_points[2]["y"] - size / 4)
        ]
    ]


def create_bones(points: List[List[Tuple[float, float, float]]]):
    armature_name = "Armature"

    bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    arm_obj = bpy.data.objects[armature_name].data

    # must be in edit mode to add bones
    index = 0

    for point in points:
        b = arm_obj.edit_bones.new(f'bone{index}')
        # a new bone will have zero length and not be kept
        # move the head/tail to keep the bone
        b.head = point[0]
        b.tail = point[1]

        index += 1

    for bone in arm_obj.edit_bones:
        if bone.name == "Bone":
            arm_obj.edit_bones.remove(bone)

    for obj in arm_obj.edit_bones:
        obj.select_head = False
        obj.select_tail = False

    return armature_name


def create_armature(cube: np.ndarray, size: int) -> str:
    return create_bones(get_bone_points(cube, size))
