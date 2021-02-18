import cv2
import numpy as np
import process_images
from sklearn.svm import SVC
import joblib

CV_PI = 3.1415926535897932384626433832795  # define


def get_board(img, model):
    font = cv2.FONT_HERSHEY_SIMPLEX
    ratio2 = 3
    kernel_size = 3
    low_threshold = 30

    clf = joblib.load(model)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (1, 1))
    edges = cv2.Canny(gray, low_threshold, low_threshold * ratio2, kernel_size)
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

    if lines is not None:
        lines = lines[0]
        # lines = sorted(lines, key=lambda line: line[0])
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("crop", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # _27_6709977.jpeg - .jpeg - _139_9456064.jpeg
    img_processed = process_images.filter_sudoku('aug/_139_9456064.jpeg')
    model = 'classifier.pkl'
    get_board(img_processed, model)
