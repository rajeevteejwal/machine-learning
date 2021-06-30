import cv2 as cv
import numpy as np
import os
from glob import glob
from PIL import Image
import tensorflow as tf
#import face_recognition
from computer_vision import model_training
from keras.models import load_model

# load haar face classifier
face_classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_classifier = cv.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

IMAGE_SIZE = [224, 224]
train_path = 'datasets/train'
valid_path = 'datasets/test'
folders = glob('datasets/train/*')
NUMBER_OF_CLASS = len(folders)

# face loader
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces:
        x = x - 10
        y = y - 10
        cropped_face = img[y:y + h + 50, x:x + w + 50]

    return cropped_face


def image_collection():
    # initialize web cam
    cap = cv.VideoCapture(0)
    count = 0
    name = ''
    new = ''
    while True:
        name = input("Enter your {} Name : ".format(new))
        try:
            if len(name) > 0:
                dir_train = './datasets/train/' + name
                os.makedirs(dir_train)
                dir_test = './datasets/test/' + name
                os.makedirs(dir_test)
                break
            else:
                pass
        except FileExistsError:
            new = 'another'
            pass

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            file_name_path = ''
            face = cv.resize(face_extractor(frame), (224, 224))
            if count <= 160:
                file_name_path = './datasets/train/' + name + '/' + str(count) + '.jpg'
            else:
                file_name_path = './datasets/test/' + name + '/' + str(count) + '.jpg'

            cv.imwrite(file_name_path, face)

            cv.putText(face, str(count), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow('Frame name', face)
        else:
            print("Face not found")
            pass

        if cv.waitKey(1) == 13 or count == 200:
            break

    cap.release()
    cv.destroyAllWindows()
    print("Collecting Images process complete")


def face_recognition():
    # load model
    # model = load_model('facefeatures_new_model.h5')
    model = tf.keras.models.load_model('facefeatures_new_model.h5')
    # Doing some Face Recognition with the web cam
    video_capture = cv.VideoCapture(0)
    while True:
        _, frame = video_capture.read()
        # canvas = detect(gray, frame)
        # image, face =face_detector(frame)

        face = face_extractor(frame)
        if type(face) is np.ndarray:
            face = cv.resize(face, (224, 224))
            im = Image.fromarray(face, 'RGB')
            # Resizing into 128x128 because we trained the model with this image size.
            img_array = np.array(im)
            # Our keras model used a 4D tensor, (images x height x width x channel)
            # So changing dimension 128x128x3 into 1x128x128x3
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.cast(img_array, tf.float32)
            pred = model.predict(img_array)
            print(pred)

            name = "None matching"

            # if(pred[0][2]>0.5):
            name = folders[np.argmax(pred)].split('\\')[1]

            cv.putText(frame, name, (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv.putText(frame, "No face found", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv.destroyAllWindows()


# comment it once you collect sufficient images
# image_collection()
#model_training.train_and_save_model()
face_recognition()
