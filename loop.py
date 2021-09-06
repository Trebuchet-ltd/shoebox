import math
import cv2
from time import sleep

try:
    import bpy
    import mathutils
except ImportError:
    pass


def calculate_angles(images):
    angles = []

    for image in images:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        segmented = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(segmented)

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        image = cv2.drawContours(image, [box], 0, (255, 0, 255), 1)

        cv2.imshow(f"Axis {images.index(image)}", image)

        angles.append(rect[2] if rect[2] < 90 else 0)

    return angles


def collapse_cube(cube):
    p_sum = 0

    layer = cube.shape[1] - 1

    for i in range(cube.shape[1]):
        c_sum = np.sum(np.sum(cube[:, i, :]))

        if c_sum > p_sum:
            if p_sum != 0 and c_sum - p_sum > 35:
                layer = i
                break
        else:
            if p_sum != 0 and p_sum - c_sum > 35:
                layer = i
                break

        p_sum = c_sum

    ret = []

    for i in range(3):
        img = np.sum(cube[:, layer:, :], axis=i, dtype="uint8")
        _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
        ret.append(img)

    return ret


def set_bone_rotation(bone, rot, axis='X'):
    mat_rot = mathutils.Matrix.Rotation(math.radians(rot), 4, axis)
    bone.matrix = mat_rot


def run_in_loop(images, size, bones, initial_angles):
    cube = create_cube(images, size)

    shadows = collapse_cube(cube)
    angles = calculate_angles(shadows)

    if angles is None:
        return

    cv2.waitKey()

    set_bone_rotation(bones["bone1"], angles[0] + initial_angles[0], axis="X")
    set_bone_rotation(bones["bone1"], angles[1] + initial_angles[1], axis="Y")
    set_bone_rotation(bones["bone1"], angles[2] + initial_angles[2], axis="Z")
    set_bone_rotation(bones["bone0"], angles[1], axis="Y")

    bones["bone1"].location = (0, 0, 0)
    bones["bone0"].location = (0, 0, 0)
