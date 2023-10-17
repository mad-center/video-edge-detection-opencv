import cv2 as cv
import numpy as np


def def_lsb():
    img = cv.imread('back.png', cv.IMREAD_GRAYSCALE)
    print(img.shape)
    w, h = img.shape
    t1 = np.ones((w, h), dtype=np.uint8) * 2 ** 0
    wm = cv.bitwise_and(img, t1)
    mask = wm[:, :] > 0
    wm[mask] = 255
    print(wm)
    cv.imshow('wm', wm)
    cv.waitKey()
    cv.destroyAllWindows()

def remove_alpga():
    img = cv.imread('back.png', cv.IMREAD_COLOR)
    cv.imwrite('back_rgb.png',img)



remove_alpga()
