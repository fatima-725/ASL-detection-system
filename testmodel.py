import numpy as np
import os
import cv2
from tensorflow import keras
from matplotlib import pyplot as plt
from pathlib import Path
from keras.models import Sequential
from keras import datasets, layers, models
from keras.layers import Conv2D, Flatten, Dense, Dropout
import tensorflow as tf
IMG_WIDTH=224
IMG_HEIGHT=224


# Define the path to the training image folder
img_trainfolder = os.path.join(str(Path(__file__).parent), r"C:\Users\amber\PycharmProjects\aslwithgui\Data\\")
def create_dataset(img_trainfolder):
    xtrain = []
    class_name = []
    # Iterate through each subdirectory in the training folder
    for dir1 in os.listdir(img_trainfolder):
        for file in os.listdir(os.path.join(img_trainfolder, dir1)):
            image_path = os.path.join(img_trainfolder, dir1, file) #load the dirrectory where image files are stored
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB) #read the image from folder
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_AREA) #compressing the all the images to same number of pixels
            image = np.array(image)
            image = image.astype('float32') #converts image to float32 datatype format
            image /= 255 #rescaling the image between 0 and 1
            xtrain.append(image)
            class_name.append(dir1)
    return xtrain, class_name

# extract the image array and class name
x_tr, class_name = create_dataset(r'C:\Users\amber\PycharmProjects\aslwithgui\Data\\') # creates a new dataset
y_tr= {k: v for v, k in enumerate(np.unique(class_name))} ## Create a dictionary to map the class names to unique integer labels
print(y_tr)
ytrain= [y_tr[class_name[i]] for i in range(len(class_name))] # Convert the class names to integer labels using the dictionary

#creating testfolder, xtest, ytest
img_testfolder = os.path.join(str(Path(__file__).parent), r"C:\Users\amber\PycharmProjects\aslwithgui\testdata\\")
def create_dataset(img_testfolder):
    xtest = []
    class_name = []
    for dir1 in os.listdir(img_testfolder):
        for file in os.listdir(os.path.join(img_testfolder, dir1)):
            image_path = os.path.join(img_testfolder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            xtest.append(image)
            class_name.append(dir1)
    return xtest, class_name

# extract the image array and class name
x_tst, class_name = create_dataset(r'C:\Users\amber\PycharmProjects\aslwithgui\testdata\\')
y_tst= {k: v for v, k in enumerate(np.unique(class_name))}
print(y_tst)
ytest= [y_tst[class_name[i]] for i in range(len(class_name))]

model=tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)),
tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
#convolved with input layer to produce the inputs for output layer,
#filters represents the number of output channels after convolution has been performed, the number of filters keep increasing for abstraction,
#kernel size is the size of matrix being used to perform convolution on the image,
#strides is the stepsize
#activation is the function which the neurons perform, in this case its relu meaning if the input is positive, output is positive and if the input is negative, output is zero
tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=(5, 5), activation='relu'),
tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=(5, 5), activation='relu'),
tf.keras.layers.Flatten(),Dropout(0.2), #flatten is used converge multi-dimensional input into 1-D, dropout is used to prevent overfitting
tf.keras.layers.Dense(100, activation='relu'),
tf.keras.layers.Dense(10, activation='relu'),
tf.keras.layers.Activation('softmax') #softmax is used to convert vector of real numbers into a probability distribution
])

#'rmsprop' is variant of the gradient descent optimization algorithm, sparse_categorical_crossentropy is loss function used in neural networks for multi-class classification problems where the target labels are integers.
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training dataset and labels, and set the number of epochs to 5
history = model.fit(x=tf.cast(np.array(x_tr), tf.float64), y=tf.cast(list(map(int,ytrain)),tf.int32), epochs=5)

# Convert the training and test data to the correct data types for prediction
x1=tf.cast(np.array(x_tr), tf.float64)
y1= y=tf.cast(list(map(int,ytrain)),tf.int32)
x2=tf.cast(np.array(x_tst), tf.float64)
y2= y=tf.cast(list(map(int,ytest)),tf.int32)

x_train = x1
y_train = y1
x_test = x2
y_test = y2

#saves the model
model.save('ModelCN.h5')

#checks loss and accuracy of test data
loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)

#testing loss and accuracy plot
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label = 'loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)