
from turtle import forward
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import cv2
import glob
import pandas as pd
import os

currDir=os.path.dirname(os.path.abspath("__file__"))
[currDir,er] = currDir.split('Primer-piloto-CoppeliaSim')
IMG_GNRL_PATH = currDir + "Img_Pre_Train/"
IMG_GNRL_PATH = IMG_GNRL_PATH.replace("\\","/")
TRAIN_PATH = "train/"
VALIDATION_PATH = "validation/"
LEFT_PATH = "left/"
RIGHT_PATH = "right/"
FORWARD_PATH = "forward/"

EPOCHS = 7
BATCH_SIZE = 10
class ImageClassifier():
    def __init__(self):
        train_data_df = self.load_train()
        validations_data,validations_label = self.load_data(val = True)
        print(train_data_df.shape)
        train_data_batch = self.data_batch(data=train_data_df)
        number_steps = train_data_df.shape[0]//BATCH_SIZE
        self.model = self.create_model()
        history = self.model.fit_generator(train_data_batch, epochs = EPOCHS, steps_per_epoch = number_steps,
            validation_data = (validations_data,validations_label))




    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=(64, 64, 3)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))        
        
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(1024))
        model.add(Dense(512))
        model.add(Dense(256))
        model.add(Dense(128))
        model.add(Dense(64))
        

        model.add(Dense(3, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer='rmsprop', metrics=['accuracy'])
        return model

    def data_batch(self,data):
        n = len(data)
        steps = n//BATCH_SIZE

        batch_data = np.zeros((BATCH_SIZE,64,64,3), dtype = np.float32)
        batch_labels = np.zeros((BATCH_SIZE,3), dtype = np.float32)

        indices = np.arange(n)

        i=0
        while True:
            np.random.shuffle(indices)
            count = 0
            next_batch = indices[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]
            for j,indexL in enumerate(next_batch):
                img_name = data.iloc[indexL]['images']
                label = data.iloc[indexL]['labels']
                if label == 'LEFT':
                    label = 0
                elif label == 'RIGHT':
                    label = 1
                else:
                    label = 2
                encoded_label = to_categorical(label,num_classes=3)
                img = cv2.imread(str(img_name))
                img = cv2.resize(img,(64,64))

                if img.shape[2] == 1:
                    img = np.dstack([img,img,img])
                color_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                #se normalizan los valores de la imagen
                color_img = color_img.astype(np.float32)/255.
                batch_data[count] = color_img
                batch_labels[count] = encoded_label

                count += 1
                if count ==BATCH_SIZE -1:
                    break
            i+=1
            yield batch_data,batch_labels
            if i>=steps:
                i=0
        
    def load_train(self):
        load_left_path = IMG_GNRL_PATH + TRAIN_PATH + LEFT_PATH
        load_right_path = IMG_GNRL_PATH + TRAIN_PATH + RIGHT_PATH
        load_forward_path = IMG_GNRL_PATH + TRAIN_PATH + FORWARD_PATH
        left_cases = self.glob_custom(load_left_path,'*.jpg')
        right_cases = self.glob_custom(load_right_path,'*.jpg')
        forward_cases = self.glob_custom(load_forward_path,'*.jpg')
        train_data = []
        train_label = []
        for img in left_cases:
            train_data.append(img)
            train_label.append('LEFT')
        for img in right_cases:
             train_data.append(img)
             train_label.append('RIGHT')
        for img in forward_cases:
             train_data.append(img)
             train_label.append('FORWARD')
        data_frameTrain = pd.DataFrame(train_data)
        data_frameTrain.columns = ['images']
        data_frameTrain['labels'] = train_label
        data_frameTrain = data_frameTrain.sample(frac =1).reset_index(drop = True)
        return data_frameTrain
    def glob_custom(self,path,imgtype):
        temp_list = glob.glob(path+imgtype)
        return list(map(lambda x: x.replace("\\","/"), temp_list))
         
    def load_data(self, val):
        load_left_path = IMG_GNRL_PATH + VALIDATION_PATH + LEFT_PATH
        load_right_path = IMG_GNRL_PATH + VALIDATION_PATH + RIGHT_PATH
        load_forward_path = IMG_GNRL_PATH + VALIDATION_PATH + FORWARD_PATH
        
        left_cases = self.glob_custom(load_left_path,'*.jpeg')
        right_cases = self.glob_custom(load_right_path,'*.jpeg')
        forward_cases = self.glob_custom(load_forward_path,'*.jpeg')
        data,labels = ([] for x in range(2))
        def image_pre_processing( case):
            for img in case:
                img = cv2.imread(str(img))
                img = cv2.resize(img, (64,64))
                img = img.astype(np.float32)/255
                if case == left_cases:
                    label = to_categorical(0,num_classes=3)
                elif case == right_cases:
                    label = to_categorical(1,num_classes=3)
                else:
                    label = to_categorical(2,num_classes=3)
                data.append(img)
                labels.append(label)
            return data,labels
        image_pre_processing(left_cases)
        image_pre_processing(right_cases)
        d,l = image_pre_processing(forward_cases)
        d = np.array(d)
        l = np.array(l)
        return d,l
agent_pretrain = ImageClassifier()