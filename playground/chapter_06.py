import cv2
import numpy as np


def example_01():
    img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
    # if > 127 => 255, else reset to 0
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print("img=\n", img)
    print("t=", t)
    print("rst=\n", rst)

def example_62():
    img = cv2.imread("./images/lena_gray.bmp")
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("img", img)
    cv2.imshow("rst", rst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_63():
    img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    print("img=\n", img)
    print("t=", t)
    print("rst=\n", rst)


def example_64():
    img = cv2.imread("./images/lena_gray.bmp")
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("img", img)
    cv2.imshow("rst", rst)
    cv2.waitKey()
    cv2.destroyAllWindows()

def example_65():
    img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    print("img=\n", img)
    print("t=", t)
    print("rst=\n", rst)

def example_66():
    img = cv2.imread("./images/lena_gray.bmp")
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    cv2.imshow("img", img)
    cv2.imshow("rst", rst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_67():
    img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    print("img=\n", img)
    print("t=", t)
    print("rst=\n", rst)


def example_68():
    img = cv2.imread("./images/lena_gray.bmp")
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow("img", img)
    cv2.imshow("rst", rst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_69():
    img = np.random.randint(0, 256, size=[4, 5], dtype=np.uint8)
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    print("img=\n", img)
    print("t=", t)
    print("rst=\n", rst)

def example_610():
    img = cv2.imread("./images/lena_gray.bmp")
    t, rst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    cv2.imshow("img", img)
    cv2.imshow("rst", rst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_611():
    img = cv2.imread("./images/computer.png", 0)
    t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # cv2.ADAPTIVE_THRESH_MEAN_C：邻域所有像素点的权重值是一致的。
    athdMEAN = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 3)
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C：与邻域各个像素点到中心点的距离有关，通过高斯方程得到各个点的权重值
    athdGAUS = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
    cv2.imshow("img", img)
    cv2.imshow("thd", thd)
    cv2.imshow("athdMEAN", athdMEAN)
    cv2.imshow("athdGAUS", athdGAUS)
    cv2.waitKey()
    cv2.destroyAllWindows()

def example_612():
    img = np.zeros((5, 5), dtype=np.uint8)
    img[0:6, 0:6] = 123
    img[2:6, 2:6] = 126
    print("img=\n", img)
    t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    print("thd=\n", thd)
    t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("t2=\n", t2)
    print("otsu=\n", otsu)


def example_613():
    img = cv2.imread("./images/lena_gray.bmp", 0)
    t1, thd = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    t2, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("img", img)
    cv2.imshow("thd", thd)
    cv2.imshow("otus", otsu)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # example_01()
    # example_62()
    # example_63()
    # example_64()
    # example_65()
    # example_66()
    # example_67()
    # example_68()
    # example_69()
    # example_610()
    # example_611()
    # example_612()
    example_613()