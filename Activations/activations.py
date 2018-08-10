import math
import time
import matplotlib
from scipy.misc import imsave
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Sequential
from keras.preprocessing import image
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from keras.applications import vgg16
from keras import backend as K

BATCH_SIZE = 128
EPOCHS = 120
NUM_CLASSES = 2

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3

LEARN_RATE = 1e-7
MOMENTUM = 0.6
img_width = 64
img_height = 64
layer_names =["conv2d_1", "conv2d_2", "conv2d_3", "conv2d_4", "conv2d_5", "conv2d_6", "conv2d_7", "conv2d_8", "conv2d_9", "conv2d_10", "conv2d_11", "conv2d_12"] 
#Filter layers
FILTER_SIZE_L1 = 7
NUM_FILTERS_L1 = 128

PADDING_SIZE = 3
PADDING_COLOR = (0, 0, 0)

def compileModel():
	model = vgg16.VGG16(weights='imagenet', include_top=False)
	print('Model loaded.')

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def printFilters(model):
    layer_dict, input_img= createLayerDictionary(model)
    kept_filters = []
    for a in range(len(layer_names)):
        layer_name = layer_names[a]
        for filter_index in range(64):
            print('Processing filter %d' % filter_index)
            start_time = time.time()
            layer_output = layer_dict[layer_name].output
            if K.image_data_format() == 'channels_first':
                loss = K.mean(layer_output[:, filter_index, :, :])
            else:
                loss = K.mean(layer_output[:, :, :, filter_index])

            # we compute the gradient of the input picture wrt this loss
            grads = K.gradients(loss, input_img)[0]

            # normalization trick: we normalize the gradient
            grads = normalize(grads)

            # this function returns the loss and grads given the input picture
            iterate = K.function([input_img], [loss, grads])

            # step size for gradient ascent
            step = 1.

            # we start from a gray image with some random noise
            if K.image_data_format() == 'channels_first':
                input_img_data = np.random.random((1, 3, img_width, img_height))
            else:
                input_img_data = np.random.random((1, img_width, img_height, 3))
            input_img_data = (input_img_data - 0.5) * 20 + 128

            # we run gradient ascent for 20 steps
            for i in range(20):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                print('Current loss value:', loss_value)
                if loss_value <= 0.:
                    # some filters get stuck to 0, we can skip them
                    break

            # decode the resulting input image
            if loss_value > 0:
                img = deprocess_image(input_img_data[0])
                kept_filters.append((img, loss_value))
            end_time = time.time()
            print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
        # we will stich the best 64 filters on a 8 x 8 grid.
        n = 6

        # the filters that have the highest loss are assumed to be better-looking.
        # we will only keep the top 64 filters.
        kept_filters.sort(key=lambda x: x[1], reverse=True)
        kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
        # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
        margin = 5
        width = n * img_width + (n - 1) * margin
        height = n * img_height + (n - 1) * margin
        stitched_filters = np.zeros((width, height, 3))

        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                img, loss = kept_filters[i * n + j]
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

        # save the result to disk
        imsave('Activation_%d.png' % (a), stitched_filters)


def createLayerDictionary(model):
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    input_img = model.input
    print("layer dictionary is created")
    return layer_dict, input_img

def main():
    AR_FP = 'highestAccuracy/image14014.jpg'
    PC_FP = 'highestAccuracy/image11.jpg'
    ACTIVATION_PATH = "highestAccuracy/"
    ACTIVATION_FILENAME = "activations_layer13_PC"
    
    MODEL_WEIGHTS_PATH ='weights1_1.hdf5'

    model = compile_model(MODEL_WEIGHTS_PATH)
    print(model.summary())
    printFilters(model)
    #create_Activations(model, ACTIVATION_PATH, ACTIVATION_FILENAME, 64,128, x)

if __name__ == '__main__':
    main()
