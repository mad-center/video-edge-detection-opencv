# https://gist.github.com/endless3cross3/2c3056aebef571c6de1016b2bbf2bdbf

import cv2


# 0.33 是为了保证高阈值/低阈值=3倍
def otsu_canny(image, lowrate=0.33):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged


img_path = r'../materials/images/op-sample-1.png'
img = cv2.imread(img_path)

edged = otsu_canny(img)

cv2.imshow('img', edged)
cv2.waitKey()
cv2.destroyAllWindows()
