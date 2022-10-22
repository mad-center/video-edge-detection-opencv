import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numba import jit

np.set_printoptions(precision=3)


# credit
# https://github.com/GBL-123/scatter-animation-for-ikun/blob/main/scatter_animation.py

# 理论上，此文件代码不能被使用。因为许可证缺失。

def get_gaussian_kernel(sigma, size):
    """ 定义高斯滤波器，size 只能是奇数 """

    if size % 2 == 0:
        raise ValueError("size 只能是奇数")

    idx = np.arange(-(size - 1) / 2, (size - 1) / 2 + 1)
    # [-2. -1.  0.  1.  2.]

    x, y = np.meshgrid(idx, idx, indexing="ij")
    # [[-2. -2. -2. -2. -2.]
    #  [-1. -1. -1. -1. -1.]
    #  [0.  0.  0.  0.  0.]
    #  [1.  1.  1.  1.  1.]
    #  [2.  2.  2.  2.  2.]]

    # [[-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]
    #  [-2. -1.  0.  1.  2.]]

    # https://softwarebydefault.com/2013/06/08/calculating-gaussian-kernels/
    kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # normalization
    kernel_matrix = kernel / np.sum(kernel)

    return kernel_matrix


@jit(nopython=True)
def gaussian_filter(gaussian_kernel, gray_img):
    """ 进行高斯滤波处理 """

    filter_img = gray_img.copy()
    a = int((gaussian_kernel.shape[0] - 1) / 2)
    # gray_img.shape[0] - a 和 gray_img.shape[1] - a
    for i in range(a, gray_img.shape[0] - a):
        for j in range(a, gray_img.shape[1] - a):
            matrix = gray_img[i - a:i + a + 1, j - a:j + a + 1]
            # sum the result of multiplying matrices
            filter_img[i, j] = np.sum(matrix * gaussian_kernel)
    filter_img = filter_img.astype("uint8")

    return filter_img


@jit(nopython=True)
def adjust_direction(direction):
    """ 将梯度方向近似为某个角度 """

    # -pi/2, -pi/4 , 0, pi/4, pi/2
    angels = np.linspace(-np.pi / 2, np.pi / 2, 5)
    # direction is a radian angle
    # 计算两个平行数组的元素差的最小值，得出 index，返回近似角度值
    adj_direction = angels[np.argmin(np.abs(direction - angels))]

    return adj_direction


@jit(nopython=True)
def calculate_grad(filter_img):
    """ 计算梯度值和梯度方向 """

    sobel_matrix = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])  # sobel 算子

    grad_matrix = filter_img.copy()
    direction_matrix = np.zeros(filter_img.shape)
    for i in range(1, filter_img.shape[0] - 1):
        for j in range(1, filter_img.shape[1] - 1):
            # https://en.wikipedia.org/wiki/Sobel_operator
            # index x [0:3] => [N-3:N-1]
            dx = np.sum(filter_img[i - 1:i + 2, j - 1:j + 2] * sobel_matrix)
            dy = np.sum(filter_img[i - 1:i + 2, j - 1:j + 2] * sobel_matrix.T)  # 注意这里 dy 用的是转置矩阵
            grad = np.sqrt(dx ** 2 + dy ** 2)
            # 避免 ZeroDivisionError
            if dx == 0:
                dx = dx + 0.01
            direction = np.arctan(dy / dx)
            adj_direction = adjust_direction(direction)  # 将梯度方向近似成某一个角度

            grad_matrix[i, j] = grad
            direction_matrix[i, j] = adj_direction
    grad_matrix = grad_matrix.astype("uint8")

    return grad_matrix, direction_matrix


@jit(nopython=True)
def NMS(grad_matrix, direction_matrix):
    """ 非极大值抑制 """

    adj_grad_matrix = np.zeros(grad_matrix.shape)
    for i in range(1, grad_matrix.shape[0] - 1):
        for j in range(1, grad_matrix.shape[1] - 1):

            # 垂直方向
            if (np.abs(direction_matrix[i, j]) == np.pi / 2) | (
                    np.abs(direction_matrix[i, j]) == -np.pi / 2):  # 选梯度的比较方向
                if grad_matrix[i, j] == np.max(grad_matrix[i - 1:i + 2, j]):
                    adj_grad_matrix[i, j] = grad_matrix[i, j]

            # 水平方向
            elif direction_matrix[i, j] == 0:
                if grad_matrix[i, j] == np.max(grad_matrix[i, j - 1:j + 2]):
                    adj_grad_matrix[i, j] = grad_matrix[i, j]

            # 对角线标准方向为左上到右下。
            #   X
            #      X
            #         X

            # 对角线方向 pi/4 ↗ 比较？
            elif direction_matrix[i, j] == np.pi / 4:
                # np.flipr fliplr() 在左右方向上翻转每行的元素，列保持不变。然后现在就可以使用 np.diag 取对角线 3 个值比较了。
                if grad_matrix[i, j] == np.max(
                        np.diag(np.fliplr(grad_matrix[i - 1:i + 2, j - 1:j + 2]))
                ):
                    adj_grad_matrix[i, j] = grad_matrix[i, j]

            # 对角线方向 -pi/4 ↘ 比较？
            elif direction_matrix[i, j] == -np.pi / 4:
                # np.diag() 用于以一维数组的形式返回方阵的对角线（或非对角线）元素。
                if grad_matrix[i, j] == np.max(
                        np.diag(grad_matrix[i - 1:i + 2, j - 1:j + 2])
                ):
                    adj_grad_matrix[i, j] = grad_matrix[i, j]

    return adj_grad_matrix


def get_threshold(img):
    """ 利用图像的标准差确定双阈值，如果用其他策略可重写该函数 """

    threshold1 = np.std(img) * 2
    threshold2 = np.std(img) * 3

    return threshold1, threshold2


@jit(nopython=True)
def get_outline(adj_grad_matrix, threshold1, threshold2):
    """ 根据梯度矩阵进行双阈值筛选 """

    a = min(threshold1, threshold2)
    b = max(threshold1, threshold2)

    outline_matrix = np.zeros(adj_grad_matrix.shape)
    for i in range(1, adj_grad_matrix.shape[0] - 1):
        for j in range(1, adj_grad_matrix.shape[1] - 1):

            # 强边界，直接标记为白色
            if adj_grad_matrix[i, j] > b:
                outline_matrix[i, j] = 255

            # 位于弱边界和强边界之间，需要判定是否和强边缘连接，如果连接则保留。
            elif adj_grad_matrix[i, j] >= a:
                # 如果以它为中心的 3x3 邻域中（这些点都和它直接相邻），存在一个点，大于高阈值的话，那么则保留。
                if np.max(adj_grad_matrix[i - 1:i + 2, j - 1:j + 2]) > b:
                    outline_matrix[i, j] = 255

    return outline_matrix


def get_img_outline(gaussian_kernel, gray_img):
    """ 获取灰度图像的轮廓，使用 canny 算法 """

    # 高斯滤波
    filter_img = gaussian_filter(gaussian_kernel, gray_img)
    # filter_img shape: (720, 1280)

    # 计算梯度和梯度方向
    grad_matrix, direction_matrix = calculate_grad(filter_img)
    # grad_matrix shape: (720, 1280); direction_matrix shape: (720, 1280)

    # 非极大值抑制：消除边误检（本来不是但检测出来是）
    adj_grad_matrix = NMS(grad_matrix, direction_matrix)
    # adj_grad_matrix shape: (720, 1280)

    # 双阈值筛选边缘：筛选边缘，得到最终的边缘
    threshold1, threshold2 = get_threshold(adj_grad_matrix)
    img_outline = get_outline(adj_grad_matrix, threshold1, threshold2)
    # img_outline shape: (720, 1280)

    return img_outline


def get_ouline_data(img_outline):
    """ 获取轮廓像素的坐标，将灰度二阶数组转化为适应 matplotlib 绘图的二维坐标数组 """

    # img_outline shape: (720, 1280)

    # This array will have shape (N, a.ndim) where N is the number of non-zero items
    # [[y0,x0],
    #  [y1,x1],
    #   ...
    #  [y(n-1),x(n-1)]]
    # 这些点 [yi,xi] 就是 255 白色像素的坐标。
    idx = np.argwhere(img_outline == 255)
    # [[106 610]
    #  [106 611]
    #  [106 612]
    #  ...
    #  [718 746]
    #  [718 798]
    #  [718 885]]

    coordinate_data = idx.copy()
    coordinate_data[:, 0] = idx[:, 1]
    # when img_outline shape is (720, 1280), img_outline.shape[0] - 1 = 719
    # 719 - yi => invert yi horizontally
    coordinate_data[:, 1] = img_outline.shape[0] - 1 - idx[:, 0]

    # coordinate_data shape: (8702, 2)

    return coordinate_data


def make_animation(img_data):
    """ 根据每一帧的坐标数据制作动画 """

    fig, ax = plt.subplots(figsize=(16, 9))
    # "o" means circle style, "ms" means the marker size in points.
    line, = ax.plot([], [], "o", ms=1, c="black")
    ax.set_xlim(0, 1279)
    ax.set_ylim(0, 719)

    def init():
        line.set_data([], [])
        return line,

    def update(n):
        # n 代表第 0,1,2... 张图片
        x = img_data[n][:, 0]
        y = img_data[n][:, 1]
        ax.set_xlim(0, 1279)
        ax.set_ylim(0, 719)
        line.set_data(x, y)
        return line,

    ani = FuncAnimation(
        fig,
        func=update,
        frames=len(img_data),
        interval=1 / 25 * 1000,
        init_func=init,
        blit=True
    )

    # ani.save("animation.mp4", fps=25)
    plt.show()


def main():
    # video_path = "./materials/videos/ikun.mp4"
    video_path = "./materials/videos/2007-autumn-anime-spot-op.mp4"
    video = cv.VideoCapture(video_path)  # 读取视频

    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)
    total_frames = video.get(cv.CAP_PROP_FRAME_COUNT)
    print(f'width: {width}; height: {height}; fps: {fps}; total_frames: {total_frames}.')

    img_data = []  # 用来存放每一张图片的轮廓数据

    # sigma can set value 1.4 for weight.
    gaussian_kernel = get_gaussian_kernel(1, 5)

    i = 1
    while video.isOpened():  # 逐帧提取图片的轮廓数据，用来画散点图
        ret, img = video.read()
        # 这里有一点需要特别注意，此时img的shape形状的width和height维度顺序交换了。
        # img shape: (720, 1280, 3)

        # for debug only, preview top 50 frames
        if i > 100:
            break

        if not ret:
            break

        print("正在解析第 {} 帧图像...".format(i))
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # gray_img shape: (720, 1280)
        img_outline = get_img_outline(gaussian_kernel, gray_img)
        coordinate_data = get_ouline_data(img_outline)
        img_data.append(coordinate_data)
        i += 1

    video.release()
    print("正在生成动画...")
    make_animation(img_data)
    input("动画生成成功，按回车键退出。")


if __name__ == "__main__":
    main()
