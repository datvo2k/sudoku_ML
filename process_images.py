import cv2
import numpy as np


def find_puzzle(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    # thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(blurred)
    lap = cv2.Laplacian(thresh, cv2.CV_64F)
    cv2.imshow('Puzzle', lap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__' :
    # _27_6709977.jpeg - _0_926439.jpeg - _139_9456064.jpeg
    find_puzzle('aug/_139_9456064.jpeg')