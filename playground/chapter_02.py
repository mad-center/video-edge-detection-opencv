import cv2
import numpy as np


def test_numpy_8x8_image():
    img = np.zeros((8, 8), dtype=np.uint8)
    print("img=\n", img)
    cv2.imshow("one", img)
    print("读取像素点 img[0,3]=", img[0, 3])
    img[0, 3] = 255
    print("修改后 img=\n", img)
    print("读取修改后像素点 img[0,3]=", img[0, 3])
    cv2.imshow("two", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def exmple_22():
    img = cv2.imread("./images/lena_gray.bmp", 0)
    cv2.imshow("before", img)
    for i in range(10, 100):  # 10...99, 90 rows
        for j in range(80, 100):  # 80...99, 20 cols
            img[i, j] = 255
    cv2.imshow("after", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def exmple_23():
    # -----------蓝色通道值--------------
    blue = np.zeros((300, 300, 3), dtype=np.uint8)
    blue[:, :, 0] = 255
    print("blue=\n", blue)
    cv2.imshow("blue", blue)
    # -----------绿色通道值--------------
    green = np.zeros((300, 300, 3), dtype=np.uint8)
    green[:, :, 1] = 255
    print("green=\n", green)
    cv2.imshow("green", green)
    # -----------红色通道值--------------
    red = np.zeros((300, 300, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    print("red=\n", red)
    cv2.imshow("red", red)
    # -----------释放窗口--------------
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_24():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, 0:100, 0] = 255  # 0...99 col ranges Blue
    img[:, 100:200, 1] = 255  # 100...199 col ranges Green
    img[:, 200:300, 2] = 255  # 200...299 col ranges Red
    print("img=\n", img)
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_25():
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    print("img=\n", img)
    print("读取像素点 img[0,3]=", img[0, 3])
    print("读取像素点 img[1,2,2]=", img[1, 2, 2])
    img[0, 3] = 255
    img[0, 0] = [66, 77, 88]
    img[1, 1, 1] = 3
    img[1, 2, 2] = 4
    img[0, 2, 0] = 5
    print("修改后 img\n", img)
    print("读取修改后像素点 img[1,2,2]=", img[1, 2, 2])


def example_26():
    img = cv2.imread("./images/lena_color.tiff")
    cv2.imshow("before", img)
    print("访问 img[0,0]=", img[0, 0])
    print("访问 img[0,0,0]=", img[0, 0, 0])
    print("访问 img[0,0,1]=", img[0, 0, 1])
    print("访问 img[0,0,2]=", img[0, 0, 2])
    print("访问 img[50,0]=", img[50, 0])
    print("访问 img[100,0]=", img[100, 0])
    # 区域 1
    for i in range(0, 50):
        for j in range(0, 100):
            for k in range(0, 3):
                img[i, j, k] = 255  # 白色
    # 区域 2
    for i in range(50, 100):
        for j in range(0, 100):
            img[i, j] = [128, 128, 128]  # 灰色
    # 区域 3
    for i in range(100, 150):
        for j in range(0, 100):
            img[i, j] = 0  # 黑色
    cv2.imshow("after", img)
    print("修改后 img[0,0]=", img[0, 0])
    print("修改后 img[0,0,0]=", img[0, 0, 0])
    print("修改后 img[0,0,1]=", img[0, 0, 1])
    print("修改后 img[0,0,2]=", img[0, 0, 2])
    print("修改后 img[50,0]=", img[50, 0])
    print("修改后 img[100,0]=", img[100, 0])
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_27():
    img = np.random.randint(10, 99, size=[5, 5], dtype=np.uint8)
    print("img=\n", img)
    print("读取像素点 img.item(3,2)=", img.item(3, 2))
    img.itemset((3, 2), 255)
    print("修改后 img=\n", img)
    print("修改后像素点 img.item(3,2)=", img.item(3, 2))


def example_28():
    img = np.random.randint(0, 256, size=[256, 256], dtype=np.uint8)
    cv2.imshow("demo", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_29():
    img = cv2.imread("./images/lena_gray.bmp", 0)
    # 测试读取、修改单个像素值
    print("读取像素点 img.item(3,2)=", img.item(3, 2))
    img.itemset((3, 2), 255)
    print("修改后像素点 img.item(3,2)=", img.item(3, 2))
    # 测试修改一个区域的像素值
    cv2.imshow("before", img)
    for i in range(10, 100):
        for j in range(80, 100):
            img.itemset((i, j), 255)
    cv2.imshow("after", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_210():
    img = np.random.randint(10, 99, size=[2, 4, 3], dtype=np.uint8)
    print("img=\n", img)
    print("读取像素点 img[1,2,0]=", img.item(1, 2, 0))
    print("读取像素点 img[0,2,1]=", img.item(0, 2, 1))
    print("读取像素点 img[1,0,2]=", img.item(1, 0, 2))
    img.itemset((1, 2, 0), 255)
    img.itemset((0, 2, 1), 255)
    img.itemset((1, 0, 2), 255)
    print("修改后 img=\n", img)
    print("修改后像素点 img[1,2,0]=", img.item(1, 2, 0))
    print("修改后像素点 img[0,2,1]=", img.item(0, 2, 1))
    print("修改后像素点 img[1,0,2]=", img.item(1, 0, 2))


def example_211():
    img = np.random.randint(0, 256, size=[256, 256, 3], dtype=np.uint8)
    cv2.imshow("demo", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_212():
    img = cv2.imread("./images/lena_color.tiff")
    cv2.imshow("before", img)
    print("访问 img.item(0,0,0)=", img.item(0, 0, 0))
    print("访问 img.item(0,0,1)=", img.item(0, 0, 1))
    print("访问 img.item(0,0,2)=", img.item(0, 0, 2))
    for i in range(0, 50):
        for j in range(0, 100):
            for k in range(0, 3):
                img.itemset((i, j, k), 255)  # 白色
    cv2.imshow("after", img)
    print("修改后 img.item(0,0,0)=", img.item(0, 0, 0))
    print("修改后 img.item(0,0,1)=", img.item(0, 0, 1))
    print("修改后 img.item(0,0,2)=", img.item(0, 0, 2))
    cv2.waitKey()


def example_213():
    a = cv2.imread("./images/lena_color.tiff", cv2.IMREAD_UNCHANGED)
    face = a[220:400, 250:350]
    cv2.imshow("original", a)
    cv2.imshow("face", face)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_214():
    a = cv2.imread("./images/lena_color.tiff", cv2.IMREAD_UNCHANGED)
    cv2.imshow("original", a)
    face = np.random.randint(0, 256, (180, 100, 3))
    a[220:400, 250:350] = face
    cv2.imshow("result", a)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_215():
    lena = cv2.imread("./images/lena_gray.bmp", cv2.IMREAD_UNCHANGED)
    dollar = cv2.imread("./images/dollar-2.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("lena", lena)
    cv2.imshow("dollar", dollar)
    face = lena[220:400, 250:350]
    dollar[160:340, 200:300] = face
    cv2.imshow("result", dollar)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_216():
    lena = cv2.imread("./images/lena_color.tiff")
    cv2.imshow("lena1", lena)
    b = lena[:, :, 0]
    g = lena[:, :, 1]
    r = lena[:, :, 2]
    cv2.imshow("b", b)
    cv2.imshow("g", g)
    cv2.imshow("r", r)

    lena[:, :, 0] = 0
    cv2.imshow("lenab0", lena)
    lena[:, :, 1] = 0
    cv2.imshow("lenab0g0", lena)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_217():
    lena = cv2.imread("./images/lena_color.tiff")
    b, g, r = cv2.split(lena)
    print(b)
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_218():
    lena = cv2.imread("./images/lena_color.tiff")
    b, g, r = cv2.split(lena)
    bgr = cv2.merge([b, g, r])
    rgb = cv2.merge([r, g, b])
    cv2.imshow("lena", lena)
    cv2.imshow("bgr", bgr)
    cv2.imshow("rgb", rgb)  # R-> B; G -> G; B -> R
    cv2.waitKey()
    cv2.destroyAllWindows()

def example_219():
    gray = cv2.imread("./images/lena_gray.bmp", 0)
    color = cv2.imread("./images/lena_color.tiff")
    print("图像 gray 属性：")
    print("gray.shape=", gray.shape)
    print("gray.size=", gray.size)
    print("gray.dtype=", gray.dtype)
    print("图像 color 属性：")
    print("color.shape=", color.shape)
    print("color.size=", color.size)
    print("color.dtype=", color.dtype)


if __name__ == '__main__':
    # test_numpy_8x8_image()
    # exmple_22()
    # exmple_23()
    # example_24()
    # example_25()
    # example_26()
    # example_27()
    # example_28()
    # example_29()
    # example_210()
    # example_211()
    # example_212()
    # example_213()
    # example_214()
    # example_215()
    # example_216()
    # example_217()
    # example_218()
    example_219()
