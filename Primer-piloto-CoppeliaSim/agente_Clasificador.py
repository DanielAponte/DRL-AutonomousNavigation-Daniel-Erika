
class ImageClassifier():
    def __init__(self):


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
        

        model.add(Dense(len(env.actions), activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(), metrics=['accuracy'])
        return model

    def image_pre_processing(self):

    def load_data(self):
