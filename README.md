# MNIST-SCANNER

import cv2, os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import load_img
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

def train():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255

    # -------------------------- CREATE MODEL ------------------------------

    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))

    # ----------------------------------------------------------------------

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train,y=y_train, epochs= 12)

    # ----------------------------------------------------------------------
    model.evaluate(x_test, y_test)
    return model

def test(model):
    print('\n..which camera wanna use ?\n')
    print('1). Laptop')
    print('2). IP WebCam')
    print('...any-thing else to EXIT.')
    camera = input('\nWhich Camera wanna use : ')

    if camera == '1':
        url = 0
    elif camera == '2':
        url = 'http://10.71.220.102:8080/video'
    else : url = 0
        
    photo = r'C:\Users\Vicky Kumar\Pictures\Saved Pictures\mnist.jpg'

    while True:
        try:
            video = cv2.VideoCapture(url)
        except : continue
            
        while True:
            check, frame = video.read()
            cv2.imshow('Capturing', frame)
            key = cv2.waitKey(1)
            if key == ord(' '):
                break

        video.release()
        cv2.destroyAllWindows()
        cv2.imwrite(photo, frame)

        try:
            image = cv2.imread(photo, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28,28))
            image = 255-image          #inverts image. Always gets read inverted.
            plt.imshow(image.reshape(28, 28),cmap='Greys')
            plt.show()
            pred = model.predict(image.reshape(1, 28, 28, 1), batch_size=1)
            print('=====================================\n\t>>> Predicted Digit : ', pred.argmax())

        except Exception as e:
            print('Error : {}'.format(e))
                    
if __name__ == "__main__":
    test(train())
        
print('exited, Thanks for using my MNIST Scanner !!!')
Using TensorFlow backend.
Epoch 1/1
60000/60000 [==============================] - 73s 1ms/step - loss: 0.2143 - accuracy: 0.9345
10000/10000 [==============================] - 4s 383us/step

..which camera wanna use ?

1). Laptop
2). IP WebCam
...any-thing else to EXIT.

Which Camera wanna use : 2
=====================================
	>>> Predicted Digit :  8
=====================================
