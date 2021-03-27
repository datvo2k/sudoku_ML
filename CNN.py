from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Dense, Activation, Conv2D, BatchNormalization
from keras.layers import Flatten, Dropout
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
import keras.optimizers as opt
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=2048)])
    except RuntimeError as e:
        print(e)


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


def define_model():
    '''base_network = MobileNet(input_shape=(28, 28, 1), include_top=False, weights=None)
    flat = Flatten()
    den = Dense(10, activation='softmax')

    model = keras.Sequential([base_network, flat, den])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model'''

    inputs = Input(shape=(28, 28, 1))

    # zero padding probably not required since the main digit is in the centre only
    X = ZeroPadding2D((1, 1))(inputs)

    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='MP1')(X)

    X = Conv2D(64, (3, 3), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization(axis=3, name='bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='MP2')(X)

    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu', name='fc1')(X)
    X = Dropout(0.4)(X)
    X = Dense(10, activation='softmax', name='fco')(X)

    model = Model(inputs=inputs, outputs=X, name='MNIST_Model')

    # print the model
    model.summary()

    # set up the optimizer
    adam = opt.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
    # compile the model with multiclass logloss (categorical cross-entropy) as the loss function
    # and use classification accuracy as another metric to measure
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

    return model


def show_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    return 0


def add_gaussian_noise(X_imgs):
    gaussian_noise_imgs = []
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs


def run_test_harness():
    # load dataset
    train_dir = 'mnist_data/Train'
    test_dir = 'mnist_data/Test'

    train_datagen = ImageDataGenerator(rescale=1 / 255.0, validation_split=0.30)

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(28, 28),
                                                        color_mode="grayscale",
                                                        batch_size=32, class_mode="categorical", subset='training')

    valid_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(28, 28),
                                                        color_mode="grayscale",
                                                        batch_size=32, class_mode="categorical", subset='validation')

    test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(28, 28),
                                                      color_mode="grayscale",
                                                      batch_size=1, class_mode=None, shuffle=False, seed=42)

    # define model
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    model = define_model()

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=3, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=0.0000001, verbose=1)

    callbacks_list = [reduce_lr]

    # fit model
    history = model.fit_generator(train_generator,
                                  validation_data=valid_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  validation_steps=valid_generator.n // valid_generator.batch_size,
                                  epochs=30, callbacks=callbacks_list)

    show_plot(history)
    print(history.history.keys())

    # save model
    model.save('final_model.h5')

    # score
    # probabilities = model.predict_generator(test_generator, 1000)
    # print(probabilities)
    return 0


if __name__ == '__main__':
    run_test_harness()
