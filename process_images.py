import cv2
import numpy as np
import imutils
import operator
from skimage import data, exposure, img_as_float
from numpy import asarray


def distance_between(p1, p2):
    a = int(p2[0] - p1[0])
    b = int(p2[1] - p1[1])

    return np.sqrt((a ** 2) + (b ** 2))


def filter_sudoku(image):
    global y, h, x, w
    img = cv2.imread(image)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    close = cv2.bitwise_not(close, close)

    cnts = cv2.findContours(close.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)  # grabs the appropriate tuple value
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)    # Find contour of box
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    ROI = orig[y:y + h, x:x + w]

    cv2.imshow('image', close)
    cv2.imshow('image_before', img)
    cv2.imshow("crop", ROI)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # _27_6709977.jpeg - .jpeg - _139_9456064.jpeg
    filter_sudoku('aug/_27_6709977.jpeg')
