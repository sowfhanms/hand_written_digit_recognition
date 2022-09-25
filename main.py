import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as k

#splliting dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
input_shape=(28,28,1)

#converting class in binary vectors

y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(Y_test,10)

x_train=x_train.astype("float32")
x_test=x_test.astype("float32")

x_train/=255
x_test/=255

batch_size=128
num_classes=10
epochs=10

model=Sequential()
model.add(Conv2D, kernel_size=(5,5),activation='relu',input_shape=input_shape)
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D,kernal_size=(5,5),activation='relu')
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
