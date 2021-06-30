import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Lambda
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#Image data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

IMAGE_SIZE = [224,224]
train_path = 'datasets/train'
valid_path = 'datasets/test'
folders = glob('datasets/train/*')
NUMBER_OF_CLASS = len(folders)


def create_model(input_size, num_of_categories):
    # add processing layer to the front of vgg
    vgg = VGG16(input_shape=input_size + [3], weights='imagenet', include_top=False)

    for layer in vgg.layers:
        layer.trainable = False

    # output of vgg
    x = Flatten()(vgg.output)
    prediction = Dense(num_of_categories, activation='softmax')(x)
    # create a model object
    local_model = Model(inputs=vgg.input, outputs=prediction)
    # model.summary()
    local_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return local_model


#train data gen
def generate_train_test_sets(train_directory='datasets/train', test_directory='datasets/test'):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    local_training_set = train_datagen.flow_from_directory(train_directory, target_size=(224, 224), batch_size=32,
                                                     class_mode='categorical')
    local_test_set = test_datagen.flow_from_directory(test_directory, target_size=(224, 224), batch_size=32,
                                                class_mode='categorical')
    return local_training_set, local_test_set


def train_model(local_model, local_training_set, local_test_set):
    r = local_model.fit_generator(local_training_set,
                            validation_data=local_test_set,
                            epochs=25,
                            steps_per_epoch=len(local_training_set),
                            validation_steps=len(local_test_set))
    return local_model


def train_and_save_model():
    model = create_model(IMAGE_SIZE, NUMBER_OF_CLASS)
    training_set, test_set = generate_train_test_sets()

    model = train_model(model, training_set, test_set)

    # model.save('.datasets/models/facefeatures_new_model.h5')
    model.save('facefeatures_new_model.h5')


#train_and_save_model()

training_set, test_set = generate_train_test_sets()
print("debugging")
