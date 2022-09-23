
from turtle import forward
from keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import cv2
import glob
import pandas as pd
import os
from datetime import date

currDir=os.path.dirname(os.path.abspath("__file__"))
[currDir,er] = currDir.split('Primer-piloto-CoppeliaSim')
IMG_GNRL_PATH = currDir + "Img_Pre_Train/"
IMG_GNRL_PATH = IMG_GNRL_PATH.replace("\\","/")
TEST_PATH = "test/"
TRAIN_PATH = "train/"
VALIDATION_PATH = "validation/"
LEFT_PATH = "left/"
RIGHT_PATH = "right/"
FORWARD_PATH = "forward/"

EPOCHS = 150
BATCH_SIZE = 10
class ImageClassifier():
    def __init__(self):
        #Variables definition
        self.labels = ['LEFT', 'RIGHT', 'FORWARD'] 
        # self.labels = ['LEFT', 'RIGHT'] 

        self.validation_data, self.validation_labels = self.load_data(data_type = 'VALIDATION')

        train_data_df = self.load_train()
        self.train_data_batch = self.data_batch(data=train_data_df)

        self.number_steps = train_data_df.shape[0]//BATCH_SIZE

        self.model = self.create_model()
        self.tensorboardCallback = TensorBoard(log_dir = "tb_logs", histogram_freq = 1)

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(256, (5, 5), input_shape=(64, 64, 3)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
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
        model.add(Dense(1024, activation=Activation('relu')))
        model.add(Dense(512, activation=Activation('relu')))
        model.add(Dense(256, activation=Activation('relu')))
        model.add(Dense(128, activation=Activation('relu')))
        model.add(Dense(64, activation=Activation('relu')))

        model.add(Dense(self.labels_number(), activation=Activation('softmax')))  # Métodos de activación disp. sigmoid o mejor softmax
        model.compile(loss= tf.keras.losses.CategoricalCrossentropy(), optimizer=Adam(learning_rate = 0.001), metrics=['accuracy'])
        return model

    def data_batch(self,data):
        n = len(data)
        steps = n//BATCH_SIZE

        batch_data = np.zeros((BATCH_SIZE,64,64,3), dtype = np.float32)
        batch_labels = np.zeros((BATCH_SIZE,self.labels_number()), dtype = np.float32)

        indices = np.arange(n)

        i=0
        while True:
            np.random.shuffle(indices)
            count = 0
            next_batch = indices[(i*BATCH_SIZE):(i+1)*BATCH_SIZE]
            for j, indexL in enumerate(next_batch):
                img_name = data.iloc[indexL]['images']
                label = self.labels.index(data.iloc[indexL]['labels'])
                encoded_label = to_categorical(label, num_classes = self.labels_number())
                img = self.read_img(img_name)
                batch_data[count] = img
                batch_labels[count] = encoded_label
            
                count += 1
                if count == BATCH_SIZE:
                    break
            i+=1
            yield batch_data, batch_labels
            if i>=steps:
                i=0
        
    def load_train(self):
        load_left_path = IMG_GNRL_PATH + TRAIN_PATH + LEFT_PATH
        load_right_path = IMG_GNRL_PATH + TRAIN_PATH + RIGHT_PATH
        load_forward_path = IMG_GNRL_PATH + TRAIN_PATH + FORWARD_PATH
        # left_cases = self.glob_custom(load_left_path,'*.jpg')
        # right_cases = self.glob_custom(load_right_path,'*.jpg')
        # forward_cases = self.glob_custom(load_forward_path,'*.jpg')
        # left_cases = self.glob_custom(load_left_path,'*.jpeg')
        # right_cases = self.glob_custom(load_right_path,'*.jpeg')
        # forward_cases = self.glob_custom(load_forward_path,'*.jpeg')
        left_cases = self.glob_custom(load_left_path,'*.jpg') + self.glob_custom(load_left_path,'*.jpeg')
        right_cases = self.glob_custom(load_right_path,'*.jpg') + self.glob_custom(load_right_path,'*.jpeg')
        forward_cases = self.glob_custom(load_forward_path,'*.jpg') + self.glob_custom(load_forward_path,'*.jpeg')
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
    
    def glob_custom(self, path, imgtype):
        temp_list = glob.glob(path+imgtype)
        return list(map(lambda x: x.replace("\\","/"), temp_list))
    
    def test_data(self, data):
        indices = np.arange(len(data)-1)
        for j,indexL in enumerate(indices):
            img_name = data.iloc[indexL]['images']
            img = cv2.imread(str(img_name))
            print('img_name: ' + str(img_name) + ' ::::: img_data: ', img)
            img = cv2.resize(img,(64,64))

    def load_data(self, data_type):
        if(data_type == 'TEST'):
            load_left_path = IMG_GNRL_PATH + TEST_PATH + LEFT_PATH
            load_right_path = IMG_GNRL_PATH + TEST_PATH + RIGHT_PATH
            load_forward_path = IMG_GNRL_PATH + TEST_PATH + FORWARD_PATH            
        else:
            load_left_path = IMG_GNRL_PATH + VALIDATION_PATH + LEFT_PATH
            load_right_path = IMG_GNRL_PATH + VALIDATION_PATH + RIGHT_PATH
            load_forward_path = IMG_GNRL_PATH + VALIDATION_PATH + FORWARD_PATH
        
        left_cases = self.glob_custom(load_left_path,'*.jpeg')
        right_cases = self.glob_custom(load_right_path,'*.jpeg')
        forward_cases = self.glob_custom(load_forward_path,'*.jpeg')
        data, labels = ([] for x in range(2))
        def image_pre_processing(case, label):
            for img in case:
                img = self.read_img(img)
                encoded_label = to_categorical(self.labels.index(label) , num_classes = self.labels_number())
                data.append(img)
                labels.append(encoded_label)
            return data,labels
        image_pre_processing(left_cases, 'LEFT')
        image_pre_processing(right_cases, 'RIGHT')
        d, l = image_pre_processing(forward_cases, 'FORWARD')
        # d, l = image_pre_processing(right_cases, 'RIGHT')
        d = np.array(d)
        l = np.array(l)
        return d,l

    def labels_number(self):
        return len(self.labels)

    def read_img(self, path):
        img = cv2.imread(str(path))
        img = cv2.resize(img,(64,64))
                
        if img.shape[2] == 1:
            img = np.dstack([img,img,img])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)/255.

    def train(self):
        history = self.model.fit(self.train_data_batch, epochs = EPOCHS, steps_per_epoch = self.number_steps,
            callbacks = [self.tensorboardCallback], verbose=2, validation_data = (self.validation_data, self.validation_labels))

    def save_model(self):
        self.model.save('img_classifier_model' + str(date.today()) + '.model')
    
    def test_model(self):
        test_data, test_label = self.load_data(data_type = 'TEST')
        predicted_data = self.model.predict(test_data, batch_size = len(test_data))
        list_result =[]
        for r in predicted_data:
            a = np.where(r == max(r)) 
            list_result.append(self.labels[int(a[0])])
        print('Predict: ', list_result)
        print('Predicted data: ', predicted_data)

agent_pretrain = ImageClassifier()


agent_pretrain.train()
agent_pretrain.test_model()
agent_pretrain.save_model()
