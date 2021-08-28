import math

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


def set_rotation(bone, rot, axis='X'):
    mat_rot = mathutils.Matrix.Rotation(math.radians(rot), 4, axis)
    bone.matrix = mat_rot


def run_in_loop(images, size, bones):
    cube = create_cube(images, size)

    shadows = collapse_cube(cube)
    angles = calculate_angles(shadows)

    if angles is None:
        return

    set_rotation(bones["bone1"], angles[0], axis="X")
    set_rotation(bones["bone1"], angles[1], axis="Z")
    set_rotation(bones["bone0"], angles[0], axis="Y")
