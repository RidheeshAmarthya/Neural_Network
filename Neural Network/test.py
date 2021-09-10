import cv2
import numpy as np
#
# image = cv2.imread('image.jpg', flags=cv2.IMREAD_COLOR)
# print(image.shape)
#
#
#
# height, width = image.shape[:2]
# tx, ty = np.array((width // 2, height // 2))
#
# T = np.array([
#     [1, 0, tx],
#     [0, 1, ty],
#     [0, 0, 1]
# ])

import cv2
import numpy as np


def translation_img(src_img, shift_distance, shape_of_out_img):
    h, w = src_img.shape[:2]
    x_distance = shift_distance[0]
    y_distance = shift_distance[1]
    ts_mat = np.array([[1, 0, x_distance], [0, 1, y_distance]])

    out_img = np.zeros(shape_of_out_img, dtype='u1')

    for i in range(h):
        for j in range(w):
            origin_x = j
            origin_y = i
            origin_xy = np.array([origin_x, origin_y, 1])

            new_xy = np.dot(ts_mat, origin_xy)
            new_x = new_xy[0]
            new_y = new_xy[1]

            if 0 < new_x < w and 0 < new_y < h:
                out_img[new_y, new_x] = src_img[i, j]
    return out_img

image = cv2.imread("image.jpg")
shift_distance = (100,100)

shifted_img = translation_img(image, shift_distance, image.shape)

cv2.imshow("input img",image)
cv2.imshow("shifted img",shifted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()