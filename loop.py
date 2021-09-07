import math
import cv2
from time import sleep

try:
    import bpy
    import mathutils
except ImportError:
    pass


def bounds_to_location(bounds):
    return mathutils.Vector((bounds[1]["x"] + bounds[1]["w"] / 2,
                             bounds[0]["y"] + bounds[0]["h"] / 2,
                             bounds[0]["x"] + bounds[0]["w"] / 2))


def calculate_angles(images):
    angles = []

    for image in images:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        segmented = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(segmented)

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
        ret.append(img)

    return ret


def set_bone_rotation(bone, rot, axis='X'):
    mat_rot = mathutils.Matrix.Rotation(math.radians(rot), 4, axis)
    bone.matrix = mat_rot


def render_image(path):
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=1)

    return cv2.imread(path, cv2.IMREAD_COLOR)


def run_in_loop(images, size, bones, initial_angles, previous_bounds):
    cube, bounds = create_cube(images, size)

    previous_location = bounds_to_location(previous_bounds)
    location = bounds_to_location(bounds)

    shadows = collapse_cube(cube)
    angles = calculate_angles(shadows)

    if angles is None:
        return previous_bounds

    bone0_loc = bones["bone0"].location + (location - previous_location)

    set_bone_rotation(bones["bone1"], angles[0] + initial_angles[0], axis="X")
    set_bone_rotation(bones["bone1"], angles[1] + initial_angles[1], axis="Y")
    set_bone_rotation(bones["bone1"], angles[2] + initial_angles[2], axis="Z")
    set_bone_rotation(bones["bone0"], angles[1], axis="Y")

    bones["bone1"].location = (0, 0, 0)
    bones["bone0"].location = bone0_loc

    cv2.imshow("rendered", render_image("/tmp/image.png"))
    cv2.waitKey()

    return bounds
