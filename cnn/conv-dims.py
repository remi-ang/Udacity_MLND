"""
Some simple tests about convolutional layers
and pooling dimensionality

"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# %% 2D Convolutional layer

filters = 32                    # number of filters
kernel_size = 3                 # height and width of the filters
input_shape = (128, 128, 3)     # depth of the previous layer

# number of parameters in the convolutional layer
nParam = filters * kernel_size**2 * input_shape[2] + filters
print("Number of parameters: {}".format(nParam))

model = Sequential()
model.add(Conv2D(filters=filters,
                 kernel_size=kernel_size,
                 strides=2,
                 padding='same',
                 activation='relu',
                 input_shape=input_shape
                 ))
model.summary()

# %% 2D Max Pooling

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()

# %% Full network

model = Sequential()

# input layer is 32x32 RGB images
# set 16 filters of size 2x2 with 'same' padding, so that the output layer has the same dimension 32x32x16
# (otherwise, with padding='valid', output dimension is 31x31x16)
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))

# MaxPooling2D reduce by 2 the dimension of the previous layer -> 16x16x16
model.add(MaxPooling2D(pool_size=2))

# double the depth, reduce dimension by 2 -> 8x8x32
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# double the depth, reduce dimension by 2: 4x4x64
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# flatten the previous layer in a 4x4x64 = 1024 vector
model.add(Flatten())

# fully connected layer ith 500 units
model.add(Dense(500, activation='relu'))

# output layer, gives a probability along 10 classes
model.add(Dense(10, activation='softmax'))

model.summary()