import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
import os
import gzip
import joblib

os.listdir()
parameter_candidates = [{'C': [0.001, 0.01, 0.1, 1, 5, 10, 100, 1000]}, ]


# predict digit using SVM and KNN


def load_mnist(filename, type, n_datapoints):
    # MNIST Images have 28*28 pixels dimension
    image_size = 28
    f = gzip.open(filename)

    if (type == 'image'):
        f.read(16)  # Skip Non-Image information
        buf = f.read(n_datapoints * image_size * image_size)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(n_datapoints, image_size * image_size)

    elif (type == 'label'):
        f.read(8)  # Skip Inessential information
        buf = f.read(n_datapoints)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        data = data.reshape(n_datapoints, 1)
    return data


# Feature scaling to [0, 1]
# x' = (x - min(x)) / (max(x) - min(x)) => x' = x / 255

print('Training')
TRAINING_SIZE = 10000  # maximun 60000

train_images = load_mnist("mnist_data/train-images-idx3-ubyte.gz", 'image', TRAINING_SIZE)
train_labels = load_mnist("mnist_data/train-labels-idx1-ubyte.gz", 'label', TRAINING_SIZE)

clf = SVC(C=5)

clf.fit(train_images, train_labels)
print("Training finished successfully")

TEST_SIZE = 1000  # 10000
test_images = load_mnist("mnist_data/t10k-images-idx3-ubyte.gz", 'image', TEST_SIZE)
test_labels = load_mnist("mnist_data/t10k-labels-idx1-ubyte.gz", 'label', TEST_SIZE)

# print(clf.score(test_images, test_labels))

print("PREDICT")
predict = clf.predict(test_images)

print("RESULT")
ac_score = metrics.accuracy_score(test_labels, predict)
cl_report = metrics.classification_report(test_labels, predict)
print("Score = ", ac_score)
print(cl_report)


# save model
joblib.dump(clf, "classifier.pkl", compress=3)
