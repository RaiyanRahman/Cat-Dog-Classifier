# Filename: classify_cat_dog.py
# Author: Raiyan Rahman
# Date: July 21st, 2019
# This is a neural network model which can classify whether a given image is
# a cat or a dog.

import cv2
import tensorflow as tf


def classify(filepath: str) -> str:
    """
    Classify the image at the given filepath as either a cat or a dog.
    :param filepath: The image filepath.
    :return: The classification.
    """
    categories = ['Dog', 'Cat']
    img_size = 50   # MUST BE THE SAME AS THE MODEL READ.

    # Read the image and get it ready for a prediction.
    img_list = prepare(filepath, img_size)

    # If an image was not successfully read.
    if len(img_list) is 0:
        return 'Please try again.'

    # Load in the model.
    model = tf.keras.models.load_model('cat_dog_classifier.model')

    # Make the prediction.
    prediction = model.predict([img_list])

    # Return the formatted prediction.
    prediction = categories[int(prediction[0][0])]
    return 'This is a ' + prediction + '.'


def prepare(filepath: str, img_size: int) -> list:
    """
    Return the read image at the given filapath after resizing it to the given
    dimension and converting it to grayscale.
    :param filepath: The filepath of the image.
    :param img_size: The dimension of the image.
    :return: new_list: The image array.
    """
    new_list = []
    try:
        img_list = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_list = cv2.resize(img_list, (img_size, img_size))
        new_list = new_list.reshape(-1, img_size, img_size, 1)
    except:
        print(filepath + ' does not exist or could not be read.')
    return new_list


if __name__ == '__main__':
    # Ask for the filename until none is given.
    filepath = ' '
    while filepath != '':
        filepath = input('What image should I predict? ')
        if filepath != '':
            prediction = classify(filepath)
            print(prediction)
