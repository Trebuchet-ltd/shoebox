import numpy as np
import cv2
from scipy import signal

try:
    import bpy
except ImportError:
    pass


def remove_background(images, background):
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    output = []

    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image = cv2.subtract(image, background)
        _, thresh = cv2.threshold(image, 1, 1, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = None
        max_area = 0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > max_area:
                max_area = w * h
                contour = cnt

        x, y, w, h = cv2.boundingRect(contour)
        output.append(thresh[y:y + h, x:x + w])

    return output


def align_images(images):
    h, w = images[0].shape[:2]

    front = np.zeros(SIZE)
    side = np.zeros(SIZE)
    top = np.zeros(SIZE)

    img = images[0]
    front[:img.shape[0], :img.shape[1]] = img

    dh, dw = images[1].shape[:2]
    img = cv2.resize(images[1], (int((dw / dh) * h), h))
    side[:img.shape[0], :img.shape[1]] = img

    img = cv2.resize(images[1], (w, int((dw / dh) * h)))
    top[:img.shape[0], :img.shape[1]] = img

    return [front, side, top]


def read_images(image_paths):
    images = []

    for path in image_paths:
        img = cv2.imread(path)
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

    print(len(faces))

    mesh.from_pydata(vertex_array, [], faces)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type='SUBSURF')
    mat = bpy.data.materials.get("back")
    bpy.context.active_object.data.materials.append(mat)


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


def main():
    background = cv2.imread("data/background.jpg")
    background = cv2.resize(background, SIZE)
    background = cv2.GaussianBlur(background, (5, 5), 0)

    images = read_images(["data/front.png", "data/side.png", "data/top.png"])
    images = remove_background(images, background)
    images = align_images(images)

    for image in images:
        cv2.imshow(f"{image}", image)

    cv2.waitKey()
    cv2.destroyAllWindows()

    cube = np.tile(images[0], (SIZE[0], 1, 1))
    cube = carve_out(cube, images[1], images[2])
    vertex_dict, vertex_array = get_edges(cube)

    draw_mesh(vertex_dict, vertex_array)


if __name__ == '__main__':
    SIZE = (64, 64)
    main()
