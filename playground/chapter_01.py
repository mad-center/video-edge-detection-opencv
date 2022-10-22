import cv2 as cv


# opencv-contrib library
# tracking：基于视觉的目标跟踪模块。
# ximgpro：图像处理扩展模块。
# xobjdetect：增强 2D 目标检测模块

def test_read_image():
    img = cv.imread("../materials/images/3.jpg")
    print(img)


test_read_image()

# 三维数组，Z 深度为 3
# [[[56  22   0]
#   [56  22   0]
#   [56  22   0]
#   ...
#   [253 250 252]
#   [253 249 254]
#   [253 249 254]]
#
#  [[56  22   0]
#   [56  22   0]
#   [56  22   0]
#   ...
#   [252 249 251]
#   [253 249 254]
#   [253 249 254]]
#
#  [[56  22   0]
#   [56  22   0]
#   [56  22   0]
#   ...
#   [252 248 253]
#   [252 248 253]
#   [252 248 253]]
#
#  ...
#
#  [[58  35  33]
#   [60  38  40]
#   [73  49  59]
#   ...
#   [59  35  15]
#   [59  35  15]
#   [59  35  15]]
#
#  [[58  35  33]
#   [60  38  40]
#   [73  49  59]
#   ...
#   [59  35  15]
#   [59  35  15]
#   [59  35  15]]
#
#  [[57  34  32]
#   [60  38  40]
#   [72  48  58]
#   ...
#   [59  35  15]
#   [59  35  15]
#   [59  35  15]]]