import numpy as np
from typing import List, Tuple


def get_nonzero_bounds(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    segmented = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(segmented)

    [[right, _], [_, top], [left, _], [_, bottom]] = np.int0(cv2.boxPoints(rect))

    return {"right": right, "top": top, "left": left, "bottom": bottom}


def get_mid_point(points):
    x = (points["left"] + points["right"]) / 2
    y = (points["top"] + points["bottom"]) / 2

    return {"x": x, "y": y}


def get_bone_points(cube, size):
    front = get_nonzero_bounds(cube[0, :, :])
    top = get_nonzero_bounds(cube[:, 20, :])
    side = get_nonzero_bounds(np.sum(cube, axis=1, dtype="uint8"))

    print(front.values())

    mid_points = [get_mid_point(bound) for bound in [front, side, top]]

    return [
        [
            (top["bottom"], side["top"], mid_points[2]["y"] - size / 4),
            (top["bottom"], side["bottom"], mid_points[2]["y"] - size / 4)
        ],
        [
            (top["bottom"], side["bottom"], mid_points[2]["y"] - size / 4),
            (side["right"], mid_points[0]["y"], mid_points[2]["y"] - size / 4)
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
