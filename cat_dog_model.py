# Filename: cat_dog_model.py
# Author: Raiyan Rahman
# Date: July 21st, 2019
# This is a neural network model which can classify whether a given image is
# a cat or a dog.
import os
import time
import random
import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,\
    Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


def create_cat_dog_classification_model(datadir='../Datasets/PetImages'):
    """
    Create a classifier that classifies the images at the given directory as
    either a cat or a dog.
    :param datadir: The directory containing the images.
    :return:
    """
    # Set the directory, categories, and image size.
    categories = ['Dog', 'Cat']
    img_size = 50
    name = 'cat_dog_classifier{}.model'.format(int(time.time()))

    # Get the images of the cats and dogs in greyscale.
    training_data = create_training_data(datadir, categories, img_size)
    # As the training data is in order, shuffle them.
    random.shuffle(training_data)
    X, y = preprocess(training_data, img_size)
    # Normalize the features.
    X = X/255.0

    # Create the training model.
    model = create_model(X)

    # Create the callback with tensorboard.
    tensorboard = TensorBoard(log_dir='logs/{}'.format(name))

    # Train the model.
    model.fit(X, y, batch_size=32, validation_split=0.1, epochs=10, callbacks=[tensorboard])

    # Save the model.
    model.save(name)


def create_model(X):
    """
    Return the created model for the neural network.
    :param X: The list of features.
    :return: model
    """
    # The first layer.
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # The second layer.
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # The third layer.
    # model.add(Dense(64))
    # model.add(Activation('relu'))

    # The output layer.
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def preprocess(training_data: list, img_size: int) -> (list, list):
    """
    Return the processed data before it can be sent to the neural network,
    using the given training dataset.
    First check if the features array already exists.
    :param training_data: The training dataset.
    :param img_size: The dimensions of the images.
    :return: The processed lists.
    """
    # Check if the file exists.
    try:
        X = np.load('features.npy')
        y = np.load('labels.npy')
    # The file does not exist.
    except:
        X = []  # Features.
        y = []  # Labels.

        # Split the data.
        for features, label in training_data:
            X.append(features)
            y.append(label)
        # Convert these to numpy arrays with gray images of the given dimensions.
        X = np.array(X).reshape(-1, img_size, img_size, 1)
        # Save the arrays.
        np.save('features.npy', X)
        np.save('labels.npy', y)

    return X, y


def create_training_data(dir: str, categories: list, img_size: int) -> list:
    """
    Return the training data composed of all the read images of each category
    from the given directory. The images are converted to greyscale.
    :param dir: The directory where the training data resides.
    :param categories: The different category of images.
    :param img_size: The size of the image.
    :return: training_data
    """
    training_data = []
    try:
        np.load('features.npy')
        np.load('labels.npy')
    except:
        # Open all the images and append them to the training data list with their
        # classification number.
        for category in categories:
            path = os.path.join(dir, category)
            classification = categories.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    # Resize the image to be the given dimensions.
                    img_array = cv2.resize(img_array, (img_size, img_size))
                    training_data.append([img_array, classification])
                except Exception as e:
                    print('Could not read the image: ' + img)

    return training_data


if __name__ == '__main__':
    create_cat_dog_classification_model()
