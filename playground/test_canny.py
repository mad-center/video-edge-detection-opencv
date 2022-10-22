import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def canny_no_blur_preprocessing():
    image = cv.imread("../materials/images/3.jpg", cv.IMREAD_GRAYSCALE)
    # image = cv.imread("../materials/images/3.jpg", cv.IMREAD_COLOR)
    frame = cv.Canny(image, 30, 90)

    frame = recolor_by_mask(frame)

    cv.namedWindow('canny', flags=cv.WINDOW_NORMAL)
    cv.imshow('canny', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()


def canny_with_blur_preprocessing():
    src = cv.imread("../materials/images/3.jpg", cv.IMREAD_GRAYSCALE)

    # 图像降噪：使用高斯模糊
    blurred = cv.GaussianBlur(src, (3, 3), 0)

    # 计算图像梯度
    xgrad = cv.Sobel(blurred, cv.CV_16SC1, 1, 0)
    ygrad = cv.Sobel(blurred, cv.CV_16SC1, 0, 1)

    # Canny 边缘检测，这里设置初始化 50 为低阈值，150 为高阈值。
    # 推荐的高低阈值比值为T2:T1=3：1或者T2：T1=2:1，其中T2为高阈值，T1为低阈值
    # 高于T2的保留，低于T1的抛弃。如果位于T1~T2之间，如果该线条和T2集合的线条有链接，那么保留；否则抛弃。
    canny = cv.Canny(xgrad, ygrad, 50, 150)
    # 直接用灰度图像
    # canny = cv.Canny(blurred, 50, 150)

    imgs = np.hstack([src, canny])
    plt.figure(figsize=(20, 10))
    plt.imshow(imgs, "gray")
    plt.axis('off')
    plt.show()

def recolor_by_mask(image):
    w, h = image.shape
    r = np.ones((w, h, 3), dtype=np.uint8) * 255
    mask_0 = image[:, :] == 0
    mask_255 = image[:, :] == 255
    r[mask_0] = [255, 255, 255]
    r[mask_255] = [255, 0, 0]
    return r

def recolor(image):
    w, h = image.shape
    # 900 x 1600 => 900 x 1600 x 3
    mat = np.ones((w, h, 3), dtype=np.uint8) * 255
    # 0 -> (255,255,255), 255 ->(255,0,0)
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


canny_no_blur_preprocessing()
