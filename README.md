# Newbie’s Deep Learning Project to Recognize Handwritten Digit
 

Developers are putting all their strength to make machines more intelligent, and smarter than humans. Deep learning is one such technique that contributes to developers enhancing machines. To memorize how a task is performed, what do humans do? Humans keep practicing and repeating that task again and again so that they get proficient in that task. After some time our brain neurons can automatically trigger and perform the task quickly and accurately. Likewise, deep learning follows the approach to solving a problem. Various types of neural network architectures are used by deep learning algorithms to solve different types of problems.

# Table of Contents
Introduction

Handwritten Digit Recognition

Prerequisites

Dataset

Steps to build Recognition System

5.1. Import libraries

5.2. Data Preprocessing

5.3. Model creation

5.4. Model training

5.5. Model Evaluation

5.6. GUI creation


# Introduction
In this article, we are going to use the MNIST dataset for the implementation of a handwritten digit recognition app. To implement this we will use a special type of deep neural network called Convolutional Neural Networks. In the end, we will also build a Graphical user interface(GUI) where you can directly draw the digit and recognize it straight away.

What is Handwritten Digit Recognition?
Handwritten digit recognition is the process to provide the ability to machines to recognize human handwritten digits. It is not an easy task for the machine because handwritten digits are not perfect, vary from person-to-person, and can be made with many different flavors.

Prerequisites

Basic knowledge of deep learning with Keras library, the Tkinter library for GUI building, and Python programming are required to run this amazing project.

Commands to Install the necessary libraries for this project:

pip install numpy
pip install tensorflow
pip install keras
pip install pillow
The MNIST dataset
Among thousands of datasets available in the market, MNIST is the most popular dataset for enthusiasts of machine learning and deep learning. Above 60,000 plus training images of handwritten digits from zero to nine and more than 10,000 images for testing are present in the MNIST dataset. So, 10 different classes are in the MNIST dataset. The images of handwritten digits are shown as a matrix of 28×28 where every cell consists of a grayscale pixel value.

mnist dataset
 

# Steps to build Handwritten Digit Recognition System
1. Import libraries and dataset
At the project beginning, we import all the needed modules for training our model. We can easily import the dataset and start working on that because the Keras library already contains many datasets and MNIST is one of them. We call mnist.load_data() function to get training data with its labels and also the testing data with its labels.

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense Flatten
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
   to split the data of training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
The Data Preprocessing
Model cannot take the image data directly so we need to perform some basic operations and process the data to make it ready for our neural network. The dimension of the training data is (60000*28*28). One more dimension is needed for the CNN model so we reshape the matrix to shape (60000*28*28*1).

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
  conversion of class vectors to matrices of  binary class 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Create the model

Its time for the creation of the CNN model for this Python-based data science project. A convolutional layer and pooling layers are the two wheels of a CNN model. The reason behind the success of CNN for image classification problems is its feasibility with grid structured data. We will use the Adadelta optimizer for the model compilation.

batch_size = 128
num_classes = 10
epochs = 10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
Train the model
To start the training of the model we can simply call the model.fit() function of Keras. It takes the training data, validation data, epochs, and batch size as the parameter.

The training of model takes some time. After succesful model training, we can save the weights and model definition in the ‘mnist.h5’ file.

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")
model.save('mnist.h5')
print("Saving the bot as mnist.h5")
Output

The model has successfully trained

Saving the bot as mnist.h5

Evaluate the model
To evaluate how accurate our model works, we have around 10,000 images in our dataset. In the training of the data model, we do not include the testing data that’s why it is new data for our model. Around 99% accuracy is achieved with this well-balanced MNIST dataset.

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
Output

 

training for handwritten digit recognition
 

# Create GUI to predict digits
To build an interactive window we have created a new file in GUI. In this file, you can draw digits on canvas, and by clicking a button, you can identify the digit. The Tkinter library is the part of Python standard library. Our predict_digit() method takes the picture as input and then activates the trained model to predict the digit.

After that to build the GUI for our app we have created the App class. In GUI canvas you can draw a digit by capturing the mouse event and with a button click, we hit the predict_digit() function and show the results.

Below is the full code for our guidigit_recog.py file:

from keras.models import load_model
from Tkinter import *
import Tkinter successful  as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np
model = load_model('mnist.h5')
def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping for model normalization
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=200, height=200, bg = "black", cursor="cross")
        self.label = tk.Label(self, text="Analyzing..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Searched", command =         self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Dlt", command = self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        #self.canvas.bind("", self.start_pos)
        self.canvas.bind("", self.draw_lines)
    def clear_all(self):
        self.canvas.delete("all")
    def classify_handwriting(self):
        Hd = self.canvas.winfo_id() # to fetch the handle of the canvas
        rect = win32gui.GetWindowRect(Hd) # to fetch the edges of the canvas
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')
    def draw_lines(slf, event):
        slf.x = event.x
        slf.y = event.y
        r=8
        slf.canvas.create_oval(slf.x-r, slf.y-r, slf.x + r, slf.y + r, fill='black')
app = App()
mainloop()
Output

output 
![image](https://github.com/surajmhulke/f/assets/136318267/0d79e708-ffa4-435b-96ba-a1533e038b95)

![image](https://github.com/surajmhulke/f/assets/136318267/5b7af1c1-9ee3-43c4-ae81-bf43e4e8f053)

handwritten digit recognition output

Conclusion
This project is beginner-friendly and can be used by data science newbies. We have created and deployed a successful deep learning project of digit recognition. We build the GUI for easy learning where we draw a digit on the canvas then we classify the digit and show the results.
