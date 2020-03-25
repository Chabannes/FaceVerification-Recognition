from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from fr_utils import *
from inception_blocks_v2 import *
from PIL import Image
from resizeimage import resizeimage
from random import randrange

# VERSION : KERAS 2.0.7 TENSORFLOW 1.2.1


# ----------------------------------------------------------------------------------------------------------------------

launch_face_verification = True # (FV)
launch_face_recognition = True # (FR)

#-----------------------------------------------------------------------------------------------------------------------

def triplet_loss(y_true, y_pred, alpha=0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # computing the distance(=encoding) between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # copputing the encoding between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # subtracting the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # taking the maximum of basic_loss and 0 and sum over the training examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0), axis=None)
    return loss


def verify(image_path, identity, database, model):

    # computing the encoding for the image
    encoding = img_to_encoding(image_path, model)

    # computing distance with the identity of the image
    dist = np.linalg.norm(encoding - database[identity])

    # valid test if dist < 0.7
    if dist < 0.7:
        door_open = True
    else:
        door_open = False

    return dist, door_open


def who_is_it(image_path, database, model):

    # compute the target encoding for the image
    encoding = img_to_encoding(image_path, model)

    # finds the closest encoding
    min_dist = 100  # initialisation

    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        identity = "Not in guest list"

    print(identity, min_dist)

    return min_dist, identity


def resize_image(base):

    for path in os.listdir(base):
        with Image.open(os.path.join(base, path)) as image:
            cover = resizeimage.resize_cover(image, [96, 96])
            cover.save(base + path, image.format)


def face_verification():

    fig = plt.figure(figsize=(6, 6))
    columns = 3
    rows = 2
    files = os.listdir('Guest/')
    plt.title('Our Guest List (FV)')
    for i in range(1, columns * rows + 1):
        img = mpimg.imread('Guest/' + files[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


    fig = plt.figure(figsize=(5, 5))
    columns = 6
    rows = 2
    files = os.listdir('People/')
    plt.title('People Showing Up (FV)')
    for i in range(1, columns * rows + 1):
        img = mpimg.imread('People/' + files[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    distance1 = verify("People/Stranger4.jpg", "Bush", database, FRmodel)[0]
    distance2 = verify("People/Stranger3.jpg", "Obama", database, FRmodel)[0]
    distance3 = verify("People/Bush.jpg", "Bush", database, FRmodel)[0]
    distance4 = verify("People/Obama.jpg", "Obama", database, FRmodel)[0]

    fig = plt.figure(figsize=(2, 2))
    columns = 2
    rows = 2
    files = ["People/Stranger4.jpg", "Guest/Bush.jpg", "People/Stranger3.jpg", "Guest/Obama.jpg"]
    distances = [distance1, distance2, distance3, distance4]
    plt.title('Failed Verification Example (FV)', fontsize=20)
    plt.text(0.45, 0.7, 'Access Denied \ndistance = %s' % np.round(distances[0], 2), fontsize=10)
    plt.text(0.45, 0.2, 'Access Denied \ndistance = %s' % np.round(distances[1], 2), fontsize=10)
    for i in range(1, columns * rows + 1):
        img = mpimg.imread(files[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    fig = plt.figure(figsize=(2, 2))
    columns = 2
    rows = 2
    files = ["People/Bush.jpg", "Guest/Bush.jpg", "People/Obama.jpg", "Guest/Obama.jpg"]
    plt.text(0.45, 0.7, 'Access Authorized\ndistance = %s' % np.round(distances[2], 2), fontsize=10)
    plt.text(0.45, 0.2, 'Access Authorized\ndistance = %s' % np.round(distances[3], 2), fontsize=10)
    plt.title('Sucessful Verification Example (FV)', fontsize=20)
    for i in range(1, columns * rows + 1):
        img = mpimg.imread(files[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    plt.show()


def face_recognition():

    fig = plt.figure(figsize=(6, 6))
    columns = 3
    rows = 2
    files = os.listdir('Guest/')
    plt.title('Our Guest List (FR)')
    for i in range(1, columns * rows + 1):
        img = mpimg.imread('Guest/' + files[i - 1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

    files = os.listdir('People/')
    plt.title('Is He Invited? (FR)')
    person_image = files[randrange(len(files) - 1)]
    img = mpimg.imread('People/' + person_image)
    plt.imshow(img)
    plt.show()

    person_dist, identity = who_is_it("People/" + person_image, database, FRmodel)

    if person_dist < 0.7:
        fig = plt.figure(figsize=(6, 6))
        columns = 1
        rows = 2
        files = ["People/" + person_image, "Guest/" + identity + '.jpg']
        plt.title('This person is indeed invited (FR)', fontsize=20)
        for i in range(1, columns * rows + 1):
            img = mpimg.imread(files[i - 1])
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)

    else:
        fig = plt.figure(figsize=(6, 6))
        files = os.listdir('People/')
        plt.title('No, that person is not a President! (FR)', fontsize=20)
        img = mpimg.imread('People/' + person_image)
        plt.imshow(img)

    plt.show()


FRmodel = faceRecoModel(input_shape=(3, 96, 96))

FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

resize_image('Guest/')
resize_image('People/')

database = {}
database["Macron"] = img_to_encoding("Guest/Macron.jpg", FRmodel)
database["Trump"] = img_to_encoding("Guest/Trump.jpg", FRmodel)
database["Obama"] = img_to_encoding("Guest/Obama.jpg", FRmodel)
database["Bush"] = img_to_encoding("Guest/Bush.jpg", FRmodel)
database["Sarkozy"] = img_to_encoding("Guest/Sarkozy.jpg", FRmodel)
database["Chirac"] = img_to_encoding("Guest/Chirac.jpg", FRmodel)



if launch_face_verification:
    face_verification()
if launch_face_recognition:
    face_recognition()
