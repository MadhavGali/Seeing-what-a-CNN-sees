import os
import numpy as np
from array import *
from PIL import Image
import keras.optimizers
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.utils import shuffle
from keras import applications
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
import math

BATCH_SIZE = 32
EPOCHS = 1
NUM_CLASSES = 2

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 3


LEARN_RATE = 1e-7
MOMENTUM = 0.6

#Filter layers
FILTER_SIZE_L1 = 3
NUM_FILTERS_L1 = 64

PADDING_SIZE = 1
PADDING_COLOR = (0, 0, 0)

def compile_model(weights_path):
    print("Compliling Model")
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(64,64,3)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))
    model.add(Dense(2, activation='softmax'))

    #model = applications.VGG16(weights=None, include_top=True, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    # Compile model    
    decay = LEARN_RATE/EPOCHS
    sgd = SGD(lr=LEARN_RATE, momentum=MOMENTUM, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
    model.load_weights(weights_path)
   
    #model.compile(loss='binary_crossentropy',optimizer=keras.optimizers.RMSprop(lr = 1e-7),metrics=['accuracy'])
    print("Model Compliled")
    return model 

    
def normalize(img_array):        
    for i in range(0,3):
        max_val = 0
        min_val = 100000
        for j in range(0,3):
            for k in range(0,3):
                if(img_array[j,k,i] > max_val):
                    max_val = img_array[j,k,i]
                if(img_array[j,k,i] < min_val):
                    min_val = img_array[j,k,i]
        img_array[:,:,i] = img_array[:,:,i] - min_val
        img_array[:,:,i] = img_array[:,:,i] / max_val
    return img_array

def visualize_single_filter(img_array):
    img = Image.new("RGB", (3, 3),'white')
    pix = img.load()
  
    for i in range(0, 3):
        for j in range(0, 3):
            print(img_array[i][j][0])
            if math.isnan(min(img_array[i][j][0], 255)):
                red = 0
            else:
                red = int(min(img_array[i][j][0], 255))
                
            if math.isnan(min(img_array[i][j][1], 255)):
                green  = 0
            else:
                green = int(min(img_array[i][j][1], 255))
            
            if math.isnan(min(img_array[i][j][2], 255)):
                blue  = 0
            else:
                blue = int(min(img_array[i][j][2], 255))
            
            pix[i,j] = (red, green, blue)
  
    return img


def create_filter_visualization(model, visualized_filter_path, visualized_filename, filter_size, num_filter):
    row = 0
    column = 0
    
    img_col = 16
    img_row = int(num_filter/16)
    
    print(model.summary())
    
    final_image = Image.new('RGB', ((filter_size*img_col)+(PADDING_SIZE*img_col-1), 
                                    (filter_size*img_row)+(PADDING_SIZE*img_row-1)),
                                    PADDING_COLOR)
    print("WIDTH: " + str((filter_size*img_col)+(PADDING_SIZE*img_col-1)))
    print("HEIGHT: " + str((filter_size*img_row)+(PADDING_SIZE*img_row-1)))
    
    for i in range(num_filter):
        img_array = model.layers[22].get_weights()[0]
        print(np.shape(img_array))
        v = normalize(img_array[:,:,:,i])
        v = v * 255
        single_filter = visualize_single_filter(v)
        final_image.paste(single_filter, (column, row))
        column += filter_size + PADDING_SIZE
        if(column > ((img_col-1) * (filter_size + PADDING_SIZE))):
            column = 0
            row = row + filter_size + PADDING_SIZE
            
    final_image.save(visualized_filter_path + visualized_filename + '.png')
    print("Filter Visualization Saved")            


def main():
    model = compile_model("weights1_1.hdf5")
    create_filter_visualization(model, 'filter_name', 'Filter_22', FILTER_SIZE_L1, NUM_FILTERS_L1) 

    
if __name__ == '__main__':
    main()    
