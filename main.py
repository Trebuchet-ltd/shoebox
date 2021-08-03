import numpy as np
import cv2

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
        _, thresh = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

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


def draw_mesh(data_arr, size):
    faces = []
    vets = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                if data_arr[i, j, k]:
                    vets.append([i, j, k])

    mesh = bpy.data.meshes.new("myBeautifulMesh")  # add the new mesh
    obj = bpy.data.objects.new(mesh.name, mesh)
    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    mesh.from_pydata(vets, [], faces)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_add(type='SUBSURF')
    mat = bpy.data.materials.get("back")
    bpy.context.active_object.data.materials.append(mat)


def main():
    background = cv2.imread("data/background.jpg")
    background = cv2.resize(background, SIZE)
    background = cv2.GaussianBlur(background, (5, 5), 0)

    images = read_images(["data/front.png", "data/side.png", "data/top.png"])
    images = remove_background(images, background)

    images = align_images(images)

    cube = np.tile(images[0], (SIZE[0], 1, 1))
    cube = carve_out(cube, images[1], images[2])

    draw_mesh(cube, SIZE[0])


if __name__ == '__main__':
    SIZE = (256, 256)
    main()
