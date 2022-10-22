import cv2
import numpy as np


def example_41():
    img = np.random.randint(0, 256, size=[2, 4, 3], dtype=np.uint8)
    rst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("img=\n", img)
    print("rst=\n", rst)
    print("像素点(1,0)直接计算得到的值=",
          img[1, 0, 0] * 0.114 + img[1, 0, 1] * 0.587 + img[1, 0, 2] * 0.299)
    print("像素点(1,0)使用公式 cv2.cvtColor()转换值=", rst[1, 0])
    # 像素点(1,0)直接计算得到的值= 141.765 => 四舍五入
    # 像素点(1,0)使用公式 cv2.cvtColor()转换值= 142


def example_42():
    img = np.random.randint(0, 256, size=[2, 4], dtype=np.uint8)
    rst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print("img=\n", img)
    print("rst=\n", rst)


def example_43():
    img = np.random.randint(0, 256, size=[2, 4, 3], dtype=np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    print("img=\n", img)
    print("rgb=\n", rgb)
    print("bgr=\n", bgr)


def example_44():
    lena = cv2.imread("./images/lena_color.tiff")
    gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # ==========打印 shape============
    print("lena.shape=", lena.shape)
    print("gray.shape=", gray.shape)
    print("bgr.shape=", bgr.shape)
    print(bgr)
    # ==========显示效果============
    cv2.imshow("lena", lena)
    cv2.imshow("gray", gray)
    cv2.imshow("bgr", bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_45():
    lena = cv2.imread("./images/lena_color.tiff")
    rgb = cv2.cvtColor(lena, cv2.COLOR_BGR2RGB)
    cv2.imshow("lena", lena)
    cv2.imshow("rgb", rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_46():
    # =========测试一下 OpenCV 中蓝色的 HSV 模式值=============
    imgBlue = np.zeros([1, 1, 3], dtype=np.uint8)
    imgBlue[0, 0, 0] = 255
    Blue = imgBlue
    BlueHSV = cv2.cvtColor(Blue, cv2.COLOR_BGR2HSV)
    print("Blue=\n", Blue)
    print("BlueHSV=\n", BlueHSV)
    # =========测试一下 OpenCV 中绿色的 HSV 模式值=============
    imgGreen = np.zeros([1, 1, 3], dtype=np.uint8)
    imgGreen[0, 0, 1] = 255
    Green = imgGreen
    GreenHSV = cv2.cvtColor(Green, cv2.COLOR_BGR2HSV)
    print("Green=\n", Green)
    print("GreenHSV=\n", GreenHSV)
    # =========测试一下 OpenCV 中红色的 HSV 模式值=============
    imgRed = np.zeros([1, 1, 3], dtype=np.uint8)
    imgRed[0, 0, 2] = 255
    Red = imgRed
    RedHSV = cv2.cvtColor(Red, cv2.COLOR_BGR2HSV)
    print("Red=\n", Red)
    print("RedHSV=\n", RedHSV)


def example_47():
    img = np.random.randint(0, 256, size=[5, 5], dtype=np.uint8)
    min = 100
    max = 200
    mask = cv2.inRange(img, min, max)
    print("img=\n", img)
    print("mask=\n", mask)


def example_48():
    img = np.ones([5, 5], dtype=np.uint8) * 9
    mask = np.zeros([5, 5], dtype=np.uint8)
    mask[0:3, 0] = 1
    mask[2:5, 2:4] = 1
    roi = cv2.bitwise_and(img, img, mask=mask)
    print("img=\n", img)
    print("mask=\n", mask)
    print("roi=\n", roi)


def example_49():
    opencv = cv2.imread("./images/OpenCV_Logo_with_text.png")
    hsv = cv2.cvtColor(opencv, cv2.COLOR_BGR2HSV)
    cv2.imshow('opencv', opencv)
    # =============指定蓝色值的范围=============
    minBlue = np.array([110, 50, 50])
    maxBlue = np.array([130, 255, 255])
    # 确定蓝色区域
    mask = cv2.inRange(hsv, minBlue, maxBlue)
    # 通过掩码控制的按位与运算，锁定蓝色区域
    blue = cv2.bitwise_and(opencv, opencv, mask=mask)
    cv2.imshow('blue', blue)
    # =============指定绿色值的范围=============
    minGreen = np.array([50, 50, 50])
    maxGreen = np.array([70, 255, 255])
    # 确定绿色区域
    mask = cv2.inRange(hsv, minGreen, maxGreen)
    # 通过掩码控制的按位与运算，锁定绿色区域
    green = cv2.bitwise_and(opencv, opencv, mask=mask)
    cv2.imshow('green', green)
    # =============指定红色值的范围=============
    minRed = np.array([0, 50, 50])
    maxRed = np.array([30, 255, 255])
    # 确定红色区域
    mask = cv2.inRange(hsv, minRed, maxRed)
    # 通过掩码控制的按位与运算，锁定红色区域
    red = cv2.bitwise_and(opencv, opencv, mask=mask)
    cv2.imshow('red', red)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_410():
    # 标记肤色
    pass


def example_411():
    img = cv2.imread("./images/lena_color.tiff")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    print(h.shape, s.shape, v.shape)
    v[:, :] = 255
    newHSV = cv2.merge([h, s, v])
    art = cv2.cvtColor(newHSV, cv2.COLOR_HSV2BGR)
    cv2.imshow("img", img)
    cv2.imshow("art", art)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_412():
    img = np.random.randint(0, 256, size=[2, 3, 3], dtype=np.uint8)
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    print("img=\n", img)
    print("bgra=\n", bgra)
    b, g, r, a = cv2.split(bgra)
    print("a=\n", a)
    a[:, :] = 125
    bgra = cv2.merge([b, g, r, a])
    print("bgra=\n", bgra)


def example_413():
    img = cv2.imread("./images/lena_color.tiff")
    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(bgra)
    a[:, :] = 125
    bgra125 = cv2.merge([b, g, r, a])
    a[:, :] = 0
    bgra0 = cv2.merge([b, g, r, a])
    cv2.imshow("img", img)
    cv2.imshow("bgra", bgra)
    cv2.imshow("bgra125", bgra125)
    cv2.imshow("bgra0", bgra0)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # 各个图像的 alpha 通道值虽然不同，但是在显示时是没有差别的

    cv2.imwrite("bgra.png", bgra)
    cv2.imwrite("bgra125.png", bgra125)
    cv2.imwrite("bgra0.png", bgra0)


if __name__ == '__main__':
    # example_41()
    # example_42()
    # example_43()
    example_44()
    # example_45()
    # example_46()
    # example_47()
    # example_48()
    # example_49()
    # example_411()
    # example_412()
    # example_413()
