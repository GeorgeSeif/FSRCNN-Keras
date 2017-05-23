from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, Merge, ZeroPadding2D, merge
from keras.preprocessing import image
from scipy.misc import imsave, imread, imresize, toimage
import numpy as np
import matplotlib.pyplot as plt

img_shape = (32, 32, 1)

model = Sequential()

input_img = Input(shape=(IMG_SIZE[0]/INPUT_SCALE, IMG_SIZE[1]/INPUT_SCALE, 1))

model = Conv2D(56, (5, 5), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)

model = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)

model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)
model = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)

model = Conv2D(56, (1, 1), padding='same', kernel_initializer='he_normal')(input_img)
model = PReLU()(model)

model = Conv2DTranspose(1, (9, 9), strides=(3, 3), padding='same')(model)

output_img = model

model = Model(input_img, output_img)

model.load_weights('fsrcnn_model.h5')

img = image.load_img('./patch.png', grayscale=True, target_size=(32, 32, 1))
x = image.img_to_array(img)
x = x.astype('float32') / 255
x = np.expand_dims(x, axis=0)

pred = model.predict(x)

test_img = np.reshape(pred, (32, 32))

imsave('test_img.png', test_img)