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


def filter_sudoku(image):
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

    '''ROI = orig[y:y + h, x:x + w]
    rows, cols, ch = ROI.shape
    # dst = rotate_image(ROI, 0)

    pts = screenCnt.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point has the smallest sum whereas the
    # bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # compute the difference between the points -- the top-right
    # will have the minumum difference and the bottom-left will
    # have the maximum difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # multiply the rectangle by the original ratio
    rect *= ratio

    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(ROI, M, (maxWidth, maxHeight))
    return warp'''

    puzzle = four_point_transform(img, screenCnt.reshape(4, 2))
    warped = four_point_transform(gray, screenCnt.reshape(4, 2))
    # check to see if we are visualizing the perspective transform
    # show the output warped image (again, for debugging purposes)
    # cv2.imshow("Puzzle Transform", puzzle)
    # cv2.waitKey(0)
    # return a 2-tuple of puzzle in both RGB and grayscale
    return puzzle


