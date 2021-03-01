import cv2
import numpy as np
import imutils
import operator
from skimage import data, exposure, img_as_float
from numpy import asarray
from imutils.perspective import four_point_transform


def distance_between(p1, p2):
    a = int(p2[0] - p1[0])
    b = int(p2[1] - p1[1])

    return np.sqrt((a ** 2) + (b ** 2))


def rotate_image(image, angle, center=None):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_sudoku_board(image):
    global y, h, x, w
    img = cv2.imread(image)
    ratio = img.shape[0] / 300.0
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
        x, y, w, h = cv2.boundingRect(c)  # Find contour of box
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    puzzle = four_point_transform(img, screenCnt.reshape(4, 2))
    warped = four_point_transform(gray, screenCnt.reshape(4, 2))
    # check to see if we are visualizing the perspective transform
    # show the output warped image (again, for debugging purposes)
    # cv2.imshow("Puzzle Transform", puzzle)
    # cv2.waitKey(0)
    # return a 2-tuple of puzzle in both RGB and grayscale
    return puzzle


