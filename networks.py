import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from evaluation import hter_metrics
from typing import Tuple


def build_cnn_model(image_shape:Tuple=(64, 64, 3), learning_rate:float =0.001, model_type='simple'):

    """
    Classic architecture for face images. Adam optimizer is also a classic optimizer.
    For binary classification, the loss will be the binary cross entropy.

    Parameters:
        image_shape (Tuple): A Tuple to enable Sequential to access the iamge shape (color image here).
        learning_rate (float): Float defined as argument when calling the model to define learning rate

        
    Returns:
        float: result of the metric HTER.
    """

    if model_type == 'base':
        # Base model
        model = Sequential()
        model.add(Input(shape=image_shape))
        model.add(Conv2D(36, (3, 3), activation='relu'))
        model.add(Conv2D(48, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(72, (3, 3), activation='relu'))
        model.add(Conv2D(72, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    elif model_type=='small_model':
        # 1) Less layers
        # 2) Layers are bigger to extract more features
        model = Sequential([
            Input(shape=image_shape),
            Conv2D(64, (3, 3), activation='relu'), 
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(32, activation='relu'),  
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError('Select appropriate name for model_type')

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

