import logging
import timeit

import cv2 as cv
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

src = cv.imread("../materials/images/3.jpg", cv.IMREAD_GRAYSCALE)
image = cv.Canny(src, 50, 150)


def recolor_by_array_index(image):
    w, h = image.shape
    mat = np.ones((w, h, 3), dtype=np.uint8) * 255
    for i in range(0, w):  # row
        for j in range(0, h):  # col
            pixel = image[i][j]
            if pixel == 0:
                mat[i, j] = [255, 255, 255]
            elif pixel == 255:
                mat[i, j] = [255, 0, 0]
    return mat


def recolor_by_itemset(image):
    w, h = image.shape
    mat = np.ones((w, h, 3), dtype=np.uint8) * 255
    for i in range(0, w):  # row
        for j in range(0, h):  # col
            pixel = image[i][j]
            if pixel == 0:
                mat.itemset((i, j, 0), 255)
                mat.itemset((i, j, 1), 255)
                mat.itemset((i, j, 2), 255)
            elif pixel == 255:
                mat.itemset((i, j, 0), 255)  # B
                mat.itemset((i, j, 1), 0)
                mat.itemset((i, j, 2), 0)
    return mat


def recolor_by_mask(image):
    w, h = image.shape
    r = np.ones((w, h, 3), dtype=np.uint8) * 255
    mask_0 = image[:, :] == 0
    mask_255 = image[:, :] == 255
    r[mask_0] = [255, 255, 255]
    r[mask_255] = [255, 0, 0]  # BGR mode
    return r


def run_test_suite():
    array_index_spent = timeit.timeit(setup='from __main__ import recolor_by_array_index, image',
                                      stmt='recolor_by_array_index(image)', number=10)
    print("array_index_spent:", array_index_spent)

    itemset_spent = timeit.timeit(setup='from __main__ import recolor_by_itemset, image',
                                  stmt='recolor_by_itemset(image)', number=10)
    print("itemset_spent:", itemset_spent)

    recolor_by_mask_spent = timeit.timeit(setup='from __main__ import recolor_by_mask, image',
                                          stmt='recolor_by_mask(image)', number=10)
    print("recolor_by_mask_spent:", recolor_by_mask_spent)


run_test_suite()

# array_index_spent: 51.7664517
# itemset_spent: 44.7930431
# recolor_by_mask_spent: 0.12137279999998896

# r = recolor_by_mask(image)
# cv.imshow('image', image)
# cv.imshow('r', r)
# cv.waitKey()
# cv.destroyAllWindows()
