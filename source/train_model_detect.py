import os
import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import PIL
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time

path = os.getcwd()
path = path[:-7]

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True


imageSize = 200

gestures_name = {}


# Ham xu ly anh resize ve 224x224 va chuyen ve numpy array
def process_image(path):
    img = PIL.Image.open(path)
    img = img.resize((imageSize, imageSize))
    img = np.array(img)
    return img

# Xu ly du lieu dau vao
def process_data(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32')
    X_data = np.stack((X_data,)*3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data

# Ham duuyet thu muc anh dung de train
def walk_file_tree(image_path):
    X_data = []
    y_data = []
    global gestures_name

    if not gestures_name:
        i=0
        for directory, subdirectories, files in os.walk(image_path):
            if(subdirectories):
                subdirectories.sort()

                for word in subdirectories:
                    gestures_name[word] = i
                    i+=1
    print(gestures_name)
    for word in gestures_name:
        print(word)
        for directory, subdirectories, files in os.walk(f'{image_path}/{word}'):
            for file in files:

                y_data.append(gestures_name[word])
                X_data.append(process_image(f'{image_path}/{word}/{file}'))

    print(len(X_data))
    X_data, y_data = process_data(X_data, y_data)
    return X_data, y_data

X_data, y_data = walk_file_tree(f'{path}/data_img')

def create_new_model():

    # Khoi tao model
    model1 = VGG16(weights='imagenet', include_top=False, input_shape=(imageSize, imageSize, 3))
    optimizer1 = optimizers.Adam()
    base_model = model1

    # Them cac lop ben tren
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dense(128, activation='relu', name='fc4')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', name='fc5')(x)

    predictions = Dense(len(gestures_name), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Dong bang cac lop duoi, chi train lop ben tren minh them vao
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

begin = time.time()
kf = KFold(n_splits=4,shuffle = True)
i=0
save_dir = f'{path}/saved_models/'

for train_index, test_index in kf.split(X_data):
    i+=1
    print(i)
    
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]

    
    model_checkpoint = ModelCheckpoint(filepath=save_dir+'model_'+str(i)+'.h5', save_best_only=True,
                                       monitor='val_accuracy',
                                  min_delta=0,
                                  patience=10,
                                  verbose=1,
                                  mode='auto')
    model = create_new_model()

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1,
              callbacks=[model_checkpoint])

    model = load_model(save_dir+'model_'+str(i)+'.h5')

    results = model.evaluate(X_test, y_test)
    results = dict(zip(model.metrics_names,results))
    
    VALIDATION_ACCURACY.append(results['accuracy'])

    print(results)

    VALIDATION_LOSS.append(results['loss'])
    
    clear_session()

