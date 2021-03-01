import cv2
import numpy as np
import process_images
from sklearn.svm import SVC
import joblib
import imutils
from skimage.filters import threshold_local
from imutils import contours
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=2048)])
    except RuntimeError as e:
        print(e)

CV_PI = 3.1415926535897932384626433832795  # define


def show_img(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_img_to_predict(img_path):
    # load the image
    img = cv2.resize(img_path, (28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


def trans_img(image):
    # Kernel size: +ve, odd, square
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    preprocess = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    preprocess = cv2.adaptiveThreshold(preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocess = cv2.dilate(preprocess, kernel, iterations=0)
    # we need grid edges, hence,
    # invert colors: gridlines will have non-zero pixels
    preprocess = cv2.bitwise_not(preprocess, preprocess)

    return preprocess


def get_line_board(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    warped = image
    warped1 = cv2.resize(warped, (610, 610))
    warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warp, 11, offset=10, method="gaussian")
    warp = (warp > T).astype("uint8") * 255
    th3 = cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return warped1


def grids(img, warped2):
    # print("im:",img.shape)
    img2 = img.copy()
    img = np.zeros((500, 500, 3), np.uint8)
    ratio2 = 3
    kernel_size = 3
    lowThreshold = 30
    frame = img

    img = cv2.resize(frame, (610, 610))
    for i in range(10):
        cv2.line(img, (0, (img.shape[0] // 9) * i), (img.shape[1], (img.shape[0] // 9) * i), (255, 255, 255), 3, 1)
        cv2.line(warped2, (0, (img.shape[0] // 9) * i), (img.shape[1], (img.shape[0] // 9) * i), (125, 0, 55), 3, 1)

    for j in range(10):
        cv2.line(img, ((img.shape[1] // 9) * j, 0), ((img.shape[1] // 9) * j, img.shape[0]), (255, 255, 255), 3, 1)
        cv2.line(warped2, ((img.shape[1] // 9) * j, 0), ((img.shape[1] // 9) * j, img.shape[0]), (125, 0, 55), 3, 1)

    # show_image(warped2, "grids")
    return img, warped2


def sort_digits_to_matrix(image, model):
    kernel = np.ones((4, 4), np.uint8)
    clf = keras.models.load_model(model)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)

    # Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    # Fix horizontal and vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=9)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=4)

    # Sort by top to bottom and each row by left to right
    invert = 255 - thresh
    cnts = cv2.findContours(invert, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    sudoku_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        area = cv2.contourArea(c)
        if area < 50000:
            row.append(c)
            if i % 9 == 0:
                (cnts, _) = contours.sort_contours(row, method="left-to-right")
                sudoku_rows.append(cnts)
                row = []

    # Iterate through each box
    for row in sudoku_rows:
        for c in row:
            mask = np.zeros(image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
            result = cv2.bitwise_and(image, mask)
            result[mask == 0] = 255

            cv2.imshow('result', result)
            cv2.waitKey(175)

            block = trans_img(result)
            cv2.imshow('block', block)

            img_predict = load_img_to_predict(block)
            digit = clf.predict_classes(img_predict)
            print(digit[0])


if __name__ == '__main__':
    # _27_6709977.jpeg - .jpeg - _139_9456064.jpeg
    img_processed = process_images.get_sudoku_board('aug/_27_6709977.jpeg')  # afer warp
    model = 'final_model.h5'
    # lines = get_line_board(img_processed)
    # grids_, warp2_ = grids(img_processed, lines)
    # show_image(warp2_, 'test')
    sort_digits_to_matrix(img_processed, model)
