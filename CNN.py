from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNet
import keras


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
    trainX = trainX.reshape((trainX.shape[0], 224, 224, 1))
    testX = testX.reshape((testX.shape[0], 224, 224, 1))
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
    base_network = MobileNet(input_shape=(224, 224, 1), include_top=False, weights=None)
    flat = Flatten()
    den = Dense(10, activation='softmax')

    model = keras.Sequential([base_network, flat, den])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

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


def run_test_harness():
    # load dataset
    train_dir = 'mnist_data/Train'
    test_dir = 'mnist_data/Test'

    train_datagen = ImageDataGenerator(rescale=1 / 255.0, rotation_range=20, zoom_range=0.05,
                                       width_shift_range=0.05, height_shift_range=0.05, shear_range=0.05,
                                       horizontal_flip=True, fill_mode="nearest", validation_split=0.20)

    test_datagen = ImageDataGenerator(rescale=1 / 255.0)

    train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(224, 224),
                                                        color_mode="grayscale",
                                                        batch_size=16, class_mode="categorical", subset='training',
                                                        shuffle=True, seed=42)

    valid_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(224, 224),
                                                        color_mode="grayscale",
                                                        batch_size=16, class_mode="categorical", subset='validation',
                                                        shuffle=True, seed=42)

    test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(224, 224),
                                                      color_mode="grayscale",
                                                      batch_size=1, class_mode=None, shuffle=False, seed=42)

    # define model
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
    model = define_model()
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=3, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # fit model
    history = model.fit_generator(train_generator, validation_data=train_generator,
                                  steps_per_epoch=train_generator.n // train_generator.batch_size,
                                  validation_steps=valid_generator.n // valid_generator.batch_size,
                                  epochs=10, callbacks=callbacks_list)
    show_plot(history)
    print(history.history.keys())
    # save model
    model.save('final_model.h5')

    return 0


if __name__ == '__main__':
    run_test_harness()
