import cv2
import numpy as np


def example_31():
    img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
    img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
    print("img1=\n", img1)
    print("img2=\n", img2)
    print("img1+img2=\n", img1 + img2)


def example_32():
    img1 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
    img2 = np.random.randint(0, 256, size=[3, 3], dtype=np.uint8)
    print("img1=\n", img1)
    print("img2=\n", img2)
    img3 = cv2.add(img1, img2)
    print("cv2.add(img1,img2)=\n", img3)


def example_33():
    a = cv2.imread("./images/lena_gray.bmp", 0)
    b = a
    result1 = a + b
    result2 = cv2.add(a, b)
    cv2.imshow("original", a)
    # 使用加号运算符计算图像像素值的和时，将和大于 255 的值进行了取模处理，取模后大
    # 于 255 的这部分值变得更小了，导致本来应该更亮的像素点变得更暗了，相加所得的图
    # 像看起来并不自然
    cv2.imshow("result1", result1)
    # 使用函数 cv2.add() 计算图像像素值的和时，将和大于 255 的值处理为饱和值 255。图像
    # 像素值相加后让图像的像素值增大了，图像整体变亮。
    cv2.imshow("result2", result2)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_34():
    img1 = np.ones((3, 4), dtype=np.uint8) * 100
    img2 = np.ones((3, 4), dtype=np.uint8) * 10
    gamma = 3
    # 60+50+3=113
    img3 = cv2.addWeighted(img1, 0.6, img2, 5, gamma)
    print(img3)


def example_35():
    a = cv2.imread("./images/lena_color.tiff")
    b = cv2.imread("./images/lena_gray.bmp")
    result = cv2.addWeighted(a, 0.6, b, 0.4, 0)
    cv2.imshow("lena color", a)
    cv2.imshow("lena gray", b)
    cv2.imshow("result", result)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_36():
    lena = cv2.imread("./images/lena_gray.bmp", cv2.IMREAD_UNCHANGED)
    dollar = cv2.imread("./images/dollar-2.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("lena", lena)
    cv2.imshow("dollar", dollar)
    # face1 and face2 must be the same shape
    face1 = lena[220:400, 250:350]
    face2 = dollar[160:340, 200:300]
    add = cv2.addWeighted(face1, 0.6, face2, 0.4, 0)
    dollar[160:340, 200:300] = add
    cv2.imshow("result", dollar)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 利用 255 这个掩码保留特定位置的值。
def example_37():
    a = np.random.randint(0, 255, (5, 5), dtype=np.uint8)
    b = np.zeros((5, 5), dtype=np.uint8)
    b[0:3, 0:3] = 255
    b[4, 4] = 255
    c = cv2.bitwise_and(a, b)
    print("a=\n", a)
    print("b=\n", b)
    print("c=\n", c)


def example_38():
    a = cv2.imread("./images/lena_gray.bmp", 0)
    b = np.zeros(a.shape, dtype=np.uint8)
    # 这里掩模位置是手动指定的
    b[100:400, 200:400] = 255
    b[100:500, 100:200] = 255
    c = cv2.bitwise_and(a, b)
    cv2.imshow("a", a)
    cv2.imshow("b", b)
    cv2.imshow("c", c)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_39():
    a = cv2.imread("./images/lena_gray.bmp", 1)
    b = np.zeros(a.shape, dtype=np.uint8)
    b[100:400, 200:400] = 255
    b[100:500, 100:200] = 255
    c = cv2.bitwise_and(a, b)
    print("a.shape=", a.shape)
    print("b.shape=", b.shape)
    cv2.imshow("a", a)
    cv2.imshow("b", b)
    cv2.imshow("c", c)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_310():
    img1 = np.ones((4, 4), dtype=np.uint8) * 3
    img2 = np.ones((4, 4), dtype=np.uint8) * 5
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    img3 = np.ones((4, 4), dtype=np.uint8) * 66
    print("img1=\n", img1)
    print("img2=\n", img2)
    print("mask=\n", mask)
    print("初始值 img3=\n", img3)

    img3 = cv2.add(img1, img2, mask=mask)
    print("求和后 img3=\n", img3)


def example_311():
    a = cv2.imread("./images/lena_gray.bmp", cv2.IMREAD_COLOR)
    w, h, c = a.shape
    mask = np.zeros((w, h), dtype=np.uint8)
    mask[100:400, 200:400] = 255
    mask[100:500, 100:200] = 255
    c = cv2.bitwise_and(a, a, mask=mask)
    print("a.shape=", a.shape)
    print("mask.shape=", mask.shape)
    cv2.imshow("a", a)
    cv2.imshow("mask", mask)
    cv2.imshow("c", c)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_312():
    img1 = np.ones((4, 4), dtype=np.uint8) * 3
    img2 = np.ones((4, 4), dtype=np.uint8) * 5
    print("img1=\n", img1)
    print("img2=\n", img2)
    img3 = cv2.add(img1, img2)
    print("cv2.add(img1,img2)=\n", img3)

    img4 = cv2.add(img1, 6)
    print("cv2.add(img1,6)\n", img4)

    img5 = cv2.add(6, img2)
    print("cv2.add(6,img2)=\n", img5)


def example_313():
    lena = cv2.imread("./images/lena_gray.bmp", 0)
    cv2.imshow("lena", lena)
    r, c = lena.shape
    # 其中 r 是行高，c 是列宽，8 表示共有 8 个通道。r、c 的值来
    # 源于要提取的图像的行高、列宽。矩阵 x 的 8 个通道分别用来提取灰度图像的 8 个位平面 (0-7)
    x = np.zeros((r, c, 8), dtype=np.uint8)

    # 设置提取矩阵
    for i in range(8):
        x[:, :, i] = 2 ** i

    r = np.zeros((r, c, 8), dtype=np.uint8)
    # 提取各个位平面，例如 r[:,:,0] 表示第 0 个平面
    for i in range(8):
        r[:, :, i] = cv2.bitwise_and(lena, x[:, :, i])

    # 阈值为 0，大于阈值的一律转化为 255 这个值。
    i = 7
    mask = r[:, :, i] > 0
    r[mask] = 255

    # 第 7 个平面的确很像原图
    cv2.imshow(str(i), r[:, :, i])
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_314():
    lena = cv2.imread("./images/lena_gray.bmp", 0)
    r, c = lena.shape
    key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
    encryption = cv2.bitwise_xor(lena, key)
    decryption = cv2.bitwise_xor(encryption, key)
    cv2.imshow("lena", lena)
    cv2.imshow("key", key)
    cv2.imshow("encryption", encryption)
    cv2.imshow("decryption", decryption)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_315():
    # 读取原始载体图像
    lena = cv2.imread("./images/lena_gray.bmp", 0)
    # 读取水印图像
    watermark = cv2.imread("./images/watermark.bmp", 0)
    # 将水印图像内的值 255 处理为 1，以方便嵌入
    # 后续章节会介绍使用 threshold 处理
    w = watermark[:, :] > 0
    watermark[w] = 1
    # 读取原始载体图像的 shape 值
    r, c = lena.shape
    # ============ 嵌入过程 ============
    # 生成元素值都是 254 的数组
    t254 = np.ones((r, c), dtype=np.uint8) * 254
    # 获取 lena 图像的高七位
    lenaH7 = cv2.bitwise_and(lena, t254)
    # 将 watermark 嵌入 lenaH7 内
    e = cv2.bitwise_or(lenaH7, watermark)
    # ============ 提取过程 ============
    # 生成元素值都是 1 的数组
    t1 = np.ones((r, c), dtype=np.uint8)
    # 从载体图像内提取水印图像
    wm = cv2.bitwise_and(e, t1)
    print(wm)
    # 将水印图像内的值 1 处理为 255，以方便显示
    # 后续章节会介绍使用 threshold 实现
    w = wm[:, :] > 0
    wm[w] = 255
    # ============ 显示 ============
    cv2.imshow("lena", lena)
    cv2.imshow("watermark", watermark * 255)  # 当前 watermark 内最大值为 1
    cv2.imshow("e", e)
    cv2.imshow("wm", wm)
    cv2.waitKey()
    cv2.destroyAllWindows()


def example_316():
    # 读取原始载体图像
    lena = cv2.imread("./images/lena_gray.bmp", 0)
    # 读取原始载体图像的 shape 值
    r, c = lena.shape
    mask = np.zeros((r, c), dtype=np.uint8)
    mask[220:400, 250:350] = 1
    # 获取一个 key, 打码、解码所使用的密钥
    key = np.random.randint(0, 256, size=[r, c], dtype=np.uint8)
    # ============ 获取打码脸 ============
    # 使用密钥 key 对原始图像 lena 加密
    lenaXorKey = cv2.bitwise_xor(lena, key)
    # 获取加密图像的脸部信息 encryptFace
    encryptFace = cv2.bitwise_and(lenaXorKey, mask * 255)
    # 将图像 lena 内的脸部值设置为 0，得到 noFace1
    noFace1 = cv2.bitwise_and(lena, (1 - mask) * 255)
    # 得到打码的 lena 图像
    maskFace = encryptFace + noFace1
    # ============ 将打码脸解码 ============
    # 将脸部打码的 lena 与密钥 key 进行异或运算，得到脸部的原始信息
    extractOriginal = cv2.bitwise_xor(maskFace, key)
    # 将解码的脸部信息 extractOriginal 提取出来，得到 extractFace
    extractFace = cv2.bitwise_and(extractOriginal, mask * 255)
    # 从脸部打码的 lena 内提取没有脸部信息的 lena 图像，得到 noFace2
    noFace2 = cv2.bitwise_and(maskFace, (1 - mask) * 255)
    # 得到解码的 lena 图像
    extractLena = noFace2 + extractFace
    # ============ 显示图像 ============
    cv2.imshow("lena", lena)
    cv2.imshow("mask", mask * 255)
    cv2.imshow("1-mask", (1 - mask) * 255)
    cv2.imshow("key", key)

    cv2.imshow("lenaXorKey", lenaXorKey)
    cv2.imshow("encryptFace", encryptFace)
    cv2.imshow("noFace1", noFace1)
    cv2.imshow("maskFace", maskFace)

    cv2.imshow("extractOriginal", extractOriginal)
    cv2.imshow("extractFace", extractFace)
    cv2.imshow("noFace2", noFace2)
    cv2.imshow("extractLena", extractLena)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # example_31()
    # example_32()
    # example_33()
    # example_34()
    # example_35()
    # example_36()
    # example_37()
    # example_38()
    # example_39()
    # example_310()
    # example_311()
    # example_312()
    example_313()
    # example_314()
    # example_315()
    # example_316()
