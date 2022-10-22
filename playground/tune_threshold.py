import cv2 as cv

# src = cv.imread("../materials/images/op-sample-1.png")
# src = cv.imread("../materials/images/3.jpg")
src = cv.imread("../materials/images/bad-case-2.png")

cv.namedWindow("bar", cv.WINDOW_NORMAL)

low_threshold = 0
high_threshold = 0


def on_change(value):
    global high_threshold
    if value != 0:
        high_threshold = 3 * value


# https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html
cv.createTrackbar("low_threshold", "bar", 20, 100, on_change)

# 图像降噪
src = cv.GaussianBlur(src, (3, 3), 0)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# 图像梯度
xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)

while True:
    low_threshold = cv.getTrackbarPos("low_threshold", "bar")
    # use gray as image
    # canny = cv.Canny(gray, low_threshold, high_threshold)

    # cv.Canny(dx, dy, threshold1, threshold2[, edges[, L2gradient]]
    canny = cv.Canny(xgrad, ygrad, low_threshold, high_threshold)
    cv.namedWindow('canny', 0)
    cv.resizeWindow('canny', 1980 // 2, 1080 // 2)
    cv.imshow("canny", canny)
    if cv.waitKey(10) & 0xFF == 27:
        break

cv.destroyAllWindows()

# bad-case-1 16
# bad-case-2 16
# bad-case-3 16
