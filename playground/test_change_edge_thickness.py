# https://stackoverflow.com/a/65047099
import cv2
import numpy as np


def tune_lower_upper_threshs(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    sigma = np.std(blur)
    mean = np.mean(blur)
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))
    print(lower, upper)

    edge = cv2.Canny(blur, lower, upper)
    cv2.imshow('edge detect', edge)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return edge


image = cv2.imread('../materials/images/3.jpg')


# edge = tune_lower_upper_threshs(image)
# cv2.imwrite('3_canny.jpg', edge)


# https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def test_dilation(image):
    img = cv2.imread(image, 0)
    # kernel more bigger, line more thick.
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow('dilation', dilation)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # cv2.imwrite('3_canny_edge_dilation.jpg', dilation)


image_path = './3_canny.jpg'
test_dilation(image_path)
