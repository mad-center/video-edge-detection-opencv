import cv2


def example_101():
    o = cv2.imread("./images/lena_gray.bmp", cv2.IMREAD_GRAYSCALE)
    r1 = cv2.Canny(o, 128, 200)
    r2 = cv2.Canny(o, 32, 128)  # 这个结果更好
    cv2.imshow("original", o)
    cv2.imshow("result1", r1)
    cv2.imshow("result2", r2)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # 当函数 cv2.Canny()的参数 threshold1 和 threshold2 的值较小时，能够捕获更多的边缘信息。


if __name__ == '__main__':
    example_101()
