#Behavioral Cloning Python script written by Jonas Chan for Udacity CarND Course
#Thursday, 23 March 2016; 22:32

#///<Disclaimer>
#///This file was originally written in iPython Notebook which is later exported
#///into a .py file which is then modified(commented and refactored) to improve
#///readability.
#///</Disclaimer>

#///Import dependencies
import keras
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D    
from random import randint

#///Set to true to obtain data distribution visualisation
debug = True

#Path of the driving log data file
path = 'data/driving_log.csv'

#Neural network parameters
validation_percentage = 0.2 #20%
number_of_epochs = 5

#///<Summary>
#///This function is used to import the data from the
#///the specified path, store it into arrays images and steering
#///The variable degrees specifies the data lesser than it to be filtered
#///before adding them into the array
#///<returns>
#///    arrays: images and steering
#///</returns>
#///</Summary>
def import_data(csv_file):
    images, steering = [], []
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        degrees = 10
        correction = degrees *0.01
        for row in reader:
            steering_value = float(row['steering'])
            images.append(row['center'])
            steering.append(steering_value)
            
            if (steering_value > 0.1 or steering_value < -0.1):
                
                # Add left camera images
                images.append(row['left'].strip())
                steering.append(steering_value + correction)
                
                # Add right camera images
                images.append(row['right'].strip())
                steering.append(steering_value - correction)
                
    return images, steering

#///<Summary>
#///This function is load the images and store them into the images array.
#///The measurements are also loaded and passed into measurement array to
#///ensure that the images and measurements are in sync with each other
#///<returns>
#///    arrays: images and steering
#///</returns>
#///</Summary>
def convert_path_to_images(images_raw, measurement):
    images = []
    measurements = []

    for i in range(len(images_raw)):
        measurements.append(measurement[i])
        images.append(cv2.imread("data/" + images_raw[i]))
    
    return images, measurements

#///<Summary>
#///Since the track is just a loop and the vehicle only turns left,
#///this function is used to flip the images in which a portion of random
#///flipped images is then added back into the array. A more detailed
#///explanation is described in the Readme file.
#///<returns>
#///    arrays: x_train and y_train
#///</returns>
#///</Summary>
def flip_images(images, measurements):
    augmented_images, augmented_measurements = [], []
    flipped_images, flipped_measurements = [], []

    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)

    for image, measurement in zip(images, measurements):
        flipped_images.append(cv2.flip(image, 1))
        flipped_measurements.append(measurement * -1.0)

    for i in range(int(len(flipped_measurements)/2)):
        random_int = randint(0, len(flipped_measurements) - 1)
        augmented_images.append(flipped_images[random_int])
        augmented_measurements.append(flipped_measurements[random_int])
    
    x_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    
    return x_train, y_train

#///<Summary>
#///This function imports the data and passes them through the first three
#///functions above as part of the preprocessing step
#///<returns>
#///    arrays: x_train and y_train
#///</returns>
#///</Summary>
def import_and_preprocess_images(csv_path):
    imported_images_paths, imported_measurements = import_data(csv_path)
    raw_images, steering_measurements = convert_path_to_images(imported_images_paths, imported_measurements)
    x_train, y_train = flip_images(raw_images, steering_measurements)
    return x_train, y_train

#///<Summary>
#///This function is used as part of the debugging process to determine the
#///number of imported datasets, number of training data as well as the distribution
#///of the measurement. More details in the Readme
#///</Summary>
def get_data_information(x_train, y_train):
    print("Total training images: " + str(len(x_train)))
    print("Total training steering measurements: " + str(len(y_train)))
    print("Training image size: " + str(x_train[0].shape))
    
    print("Plotting data into histogram...")
    plt.hist(y_train, bins=20)
    plt.title("Steering Angles Distribution")
    plt.xlabel("Angle")
    plt.ylabel("Frequency")
    plt.show()
          
    print("Showing random example image and steering data...")
    random_int = randint(0, len(y_train) - 1)
    plt.imshow(x_train[random_int])
    plt.title("Steering angle: " + str(y_train[random_int]))
    plt.show()

#///<Summary>
#///Start importing and preprocessing the data
#///</Summary>
x_train, y_train = import_and_preprocess_images(path)

if debug:
    get_data_information(x_train, y_train)

#///<Summary>
#///This function is used to visualise the loss and validation loss
#///from the neural network
#///</Summary>
def visualise_loss_data(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

print('Starting convolutional processing...')

#///<Summary>
#///Please see the Readme file for this.
#/// PLEASE NOTE: No generators were used since not much data were used as
#/// as part of the training process and the rig used was sufficient
#/// to conduct the training process.
#///</Summary>
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

model.add(Convolution2D(32,3,3, activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(32,3,3, activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(64,3,3, activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))

model.summary()

model.compile(loss = 'mse', optimizer = 'adam')

#///Start the model with X amount of training data and 20% of the training data for validation_split
#///and run it through for 5 times.
history_object = model.fit(x_train, y_train, validation_split = validation_percentage, shuffle = True, nb_epoch = number_of_epochs)

#///Start visualising the loss when the training is complete
visualise_loss_data(history_object)

#///Save the weights to be used in the simulation
print("Saving data into model.h5")
model.save('model.h5')
print("Model saved!")

