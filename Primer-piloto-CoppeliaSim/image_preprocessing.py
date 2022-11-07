# Image Preprocessing

# Importing the libraries
import numpy as np
import cv2

from gym.core import ObservationWrapper
from gym.spaces.box import Box

# Preprocessing the Images

class PreprocessImage(ObservationWrapper):
    
    def __init__(self, env, height = 64, width = 64, grayscale = True, crop = lambda img: img):
        super(PreprocessImage, self).__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def _observation(self, img):
        if self.grayscale:
            img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.img_size)
        
        return img
def convert(img,img_size):
    img = cv2.resize(img,img_size)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=np.array(np.transpose(img))
    img=np.expand_dims(img, axis=0)
    
    return img