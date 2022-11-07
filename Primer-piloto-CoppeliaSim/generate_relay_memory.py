from PIL import Image
from collections import deque
import cv2
import os
import json
import numpy as np
from datetime import date
from datetime import datetime

REPLAY_MEMORY_SIZE = 384

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Replay_Memory():
    def __init__(self):
        currDir=os.path.dirname(os.path.abspath("__file__"))
        [currDir,er] = currDir.split('Primer-piloto-CoppeliaSim')
        ModelPath = currDir + "Capturas_e_Imagenes/"
        self.ModelPath = ModelPath.replace("\\","/")

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        for i in range(4):
            self.load_large_route()
            self.load_short_route()

        
    def read_image(self, image_name):
        return np.array(Image.open(self.ModelPath + '/' + image_name).convert('RGB'))

    def glob_custom(self, path, imgtype):
        temp_list = glob.glob(path+imgtype)
        return list(map(lambda x: x.replace("\\","/"), temp_list))
        
    def save_replay_memory(self):        
        json_object = json.dumps(list(self.replay_memory), cls = NpEncoder)
        with open('replay_memory_pre_train' + str(date.today()) + '.json', 'w') as f:
            f.write(json_object)
    
    def load_large_route(self):
        self.replay_memory.append((self.read_image('67_67_0.jpeg'), 0, 0.7, self.read_image('67_67_90.jpeg'), False))
        self.replay_memory.append((self.read_image('67_67_90.jpeg'), 2, 0.73, self.read_image('67_52_90.jpeg'), False))
        self.replay_memory.append((self.read_image('67_52_90.jpeg'), 1, 0.73, self.read_image('67_52_0.jpeg'), False))
        self.replay_memory.append((self.read_image('67_52_0.jpeg'), 2, 0.76, self.read_image('52_52_0.jpeg'), False))
        self.replay_memory.append((self.read_image('52_52_0.jpeg'), 1, 0.76, self.read_image('52_52_270.jpeg'), False))
        self.replay_memory.append((self.read_image('52_52_270.jpeg'), 2, 0.79, self.read_image('52_67_270.jpeg'), False))
        self.replay_memory.append((self.read_image('52_67_270.jpeg'), 0, 0.79, self.read_image('52_67_0.jpeg'), False))
        self.replay_memory.append((self.read_image('52_67_0.jpeg'), 2, 0.82, self.read_image('38_67_0.jpeg'), False))
        self.replay_memory.append((self.read_image('38_67_0.jpeg'), 2, 0.85, self.read_image('24_67_0.jpeg'), False))
        self.replay_memory.append((self.read_image('24_67_0.jpeg'), 0, 0.85, self.read_image('24_67_90.jpeg'), False))
        self.replay_memory.append((self.read_image('24_67_90.jpeg'), 2, 0.88, self.read_image('24_52_90.jpeg'), False))
        self.replay_memory.append((self.read_image('24_52_90.jpeg'), 2, 0.91, self.read_image('24_38_90.jpeg'), False))
        self.replay_memory.append((self.read_image('24_38_90.jpeg'), 0, 0.91, self.read_image('24_38_180.jpeg'), False))
        self.replay_memory.append((self.read_image('24_38_180.jpeg'), 2, 0.94, self.read_image('38_38_180.jpeg'), False))
        self.replay_memory.append((self.read_image('38_38_180.jpeg'), 1, 0.94, self.read_image('38_38_90.jpeg'), False))
        self.replay_memory.append((self.read_image('38_38_90.jpeg'), 2, 0.97, self.read_image('38_24_90.jpeg'), False))
        self.replay_memory.append((self.read_image('38_24_90.jpeg'), 1, 0.97, self.read_image('38_24_0.jpeg'), False))
        self.replay_memory.append((self.read_image('38_24_0.jpeg'), 2, 1, self.read_image('24_24_0.jpeg'), True))

    def load_short_route(self):
        self.replay_memory.append((self.read_image('67_67_0.jpeg'), 0, 0.7,      self.read_image('67_67_90.jpeg' ), False))
        self.replay_memory.append((self.read_image('67_67_90.jpeg'), 2, 0.73,    self.read_image('67_52_90.jpeg' ), False))
        self.replay_memory.append((self.read_image('67_52_90.jpeg'), 1, 0.73,    self.read_image('67_52_0.jpeg'  ), False))
        self.replay_memory.append((self.read_image('67_52_0.jpeg'), 2, 0.76,     self.read_image('52_52_0.jpeg'  ), False))
        self.replay_memory.append((self.read_image('52_52_0.jpeg'), 0, 0.76,     self.read_image('52_52_90.jpeg' ), False))
        self.replay_memory.append((self.read_image('52_52_90.jpeg'), 2, 0.79,    self.read_image('52_38_90.jpeg' ), False))
        self.replay_memory.append((self.read_image('52_38_90.jpeg'), 0, 0.79,    self.read_image('52_38_180.jpeg'), False))
        self.replay_memory.append((self.read_image('52_38_180.jpeg'), 2, 0.82,   self.read_image('67_38_180.jpeg'), False))
        self.replay_memory.append((self.read_image('67_38_180.jpeg'), 1, 0.82,   self.read_image('67_38_90.jpeg' ), False))
        self.replay_memory.append((self.read_image('67_38_90.jpeg'), 2, 0.85,    self.read_image('67_24_90.jpeg' ), False))
        self.replay_memory.append((self.read_image('67_24_90.jpeg'), 1, 0.85,    self.read_image('67_24_0.jpeg'  ), False))
        self.replay_memory.append((self.read_image('67_24_0.jpeg'), 2, 0.88,     self.read_image('52_24_0.jpeg'  ), False))
        self.replay_memory.append((self.read_image('52_24_0.jpeg'), 2, 0.97,     self.read_image('38_24_0.jpeg'  ), False))
        self.replay_memory.append((self.read_image('38_24_0.jpeg'), 2, 1,        self.read_image('24_24_0.jpeg'  ), True))

replay_mem = Replay_Memory()
replay_mem.save_replay_memory()
