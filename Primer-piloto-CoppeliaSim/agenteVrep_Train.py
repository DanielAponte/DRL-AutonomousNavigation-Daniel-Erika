#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:36:20 2020

@author: daniel y erika
"""
from turtle import width
import sim                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import logging
from datetime import date
from datetime import datetime

BATCH_SIZE = 20

class Agent():   
    def __init__(self):
        self.position = 0
        self.route = 0
    def changeRoute(self, newRoute):
        self.route = newRoute
        self.position = 0
    def changePosition(self):
        self.position += 1
    def reset(self, init_pos, init_route):
        self.position = init_pos
        self.route = init_route

class Environment():   
    
    def __init__(self):

        self.reset_moods = ['static', 'random']
        self.reset_mood = 0
        self.TD = 0
        forward = 'w'
        left = 'a'
        right = 'd'
        #backward = 's'
        self.actions = [forward, right, left]
        self.position = [0,0,0]
        self.agent = Agent()

        self.list_init_positions = [
            (6.7, 6.7, 0),
            (6.7, 6.7, 90),
            (6.7, 5.2, 90),
            (6.7, 5.2, 0),
            (5.2, 5.2, 0)
        ]
        self.list_route_short = [
            (5.2, 5.2, 90),
            (5.2, 3.8, 90),
            (5.2, 3.8, 180),
            (6.7, 3.8, 180),
            (6.7, 3.8, 90),
            (6.7, 2.4, 90),
            (6.7, 2.4, 0), 
            (5.2, 2.4, 0), 
            (3.8, 2.4, 0), 
            (2.4, 2.4, 0)
        ]
        self.list_route_long = [
            (5.2, 5.2, 270),
            (5.2, 6.7, 270),
            (5.2, 6.7, 0),
            (3.8, 6.7, 0),
            (2.4, 6.7, 0),
            (2.4, 6.7, 90),
            (2.4, 5.2, 90),
            (2.4, 3.8, 90),
            (2.4, 3.8, 180),
            (3.8, 3.8, 180),
            (3.8, 3.8, 90),
            (3.8, 2.4, 90),
            (3.8, 2.4, 0),
            (2.4, 2.4, 0)
        ]
        self.list_posibles_inits = [
            (6.7, 6.7, 0),
            (6.7, 6.7, 90),
            (6.7, 5.2, 90),
            (6.7, 5.2, 0),
            (5.2, 5.2, 0),
            (5.2, 5.2, 270),
            (5.2, 6.7, 270),
            (5.2, 6.7, 0),
            (3.8, 6.7, 0),
            (2.4, 6.7, 0),
            (2.4, 6.7, 90),
            (2.4, 5.2, 90),
            (2.4, 3.8, 90),
            (2.4, 3.8, 180),
            (3.8, 3.8, 180),
            (3.8, 3.8, 90),
            (3.8, 2.4, 90),
            (5.2, 5.2, 90),
            (5.2, 3.8, 90),
            (5.2, 3.8, 180),
            (6.7, 3.8, 180),
            (6.7, 3.8, 90),
            (6.7, 2.4, 90),
            (6.7, 2.4, 0),
            (5.2, 2.4, 0)
        ]
        self.dict_img = {
            (6.7, 6.7, 0) : '67_67_0',
            (6.7, 6.7, 90) : '67_67_90',
            (6.7, 5.2, 90) : '67_52_90',
            (6.7, 5.2, 0) : '67_52_0',
            (5.2, 5.2, 0) : '52_52_0',
            (5.2, 5.2, 90) : '52_52_90',
            (5.2, 3.8, 90) : '52_38_90',
            (5.2, 3.8, 180) : '52_38_180',
            (6.7, 3.8, 180): '67_38_180',
            (6.7, 3.8, 90) : '67_38_90',
            (6.7, 2.4, 90): '67_24_90',
            (6.7, 2.4, 0) : '67_24_0',
            (5.2, 2.4, 0) : '52_24_0',
            (2.4, 2.4, 0) : '24_24_0',
            (5.2, 5.2, 270) : '52_52_270',
            (5.2, 6.7, 270) : '52_67_270',
            (5.2, 6.7, 0) : '52_67_0',
            (3.8, 6.7, 0) : '38_67_0',
            (2.4, 6.7, 0) : '24_67_0',
            (2.4, 6.7, 90) : '24_67_90',
            (2.4, 5.2, 90) : '24_52_90',
            (2.4, 3.8, 90) : '24_38_90',
            (2.4, 3.8, 180) : '24_38_180',
            (3.8, 3.8, 180) : '38_38_180',
            (3.8, 3.8, 90) : '38_38_90',
            (3.8, 2.4, 90) : '38_24_90',
            (3.8, 2.4, 0) : '38_24_0'
        }
        self.dict_posible_outcomes = {
            (6.7, 6.7, 0) : 'a;',
            (6.7, 6.7, 90) : 'w;',
            (6.7, 5.2, 90) : 'd;',
            (6.7, 5.2, 0) : 'w;',
            (5.2, 5.2, 0) : 'a;d',
            (5.2, 5.2, 270) : 'w;',
            (5.2, 6.7, 270) : 'a;',
            (5.2, 6.7, 0) : 'w;',
            (3.8, 6.7, 0) : 'w;',
            (2.4, 6.7, 0) : 'a;',
            (2.4, 6.7, 90) : 'w;',
            (2.4, 5.2, 90) : 'w;',
            (2.4, 3.8, 90) : 'a;',
            (2.4, 3.8, 180) : 'w;',
            (3.8, 3.8, 180) : 'd;',
            (3.8, 3.8, 90) : 'w;',
            (3.8, 2.4, 90) : 'd;',
            (3.8, 2.4, 0) : 'w;',
            (5.2, 5.2, 90) : 'w;',
            (5.2, 3.8, 90) : 'a;',
            (5.2, 3.8, 180) : 'w;',
            (6.7, 3.8, 180): 'd;',
            (6.7, 3.8, 90) : 'w;',
            (6.7, 2.4, 90): 'd;',
            (6.7, 2.4, 0) : 'w;',
            (5.2, 2.4, 0) : 'w;',
            (2.4, 2.4, 0) : 'w;'
        }
        
        self.routes = [self.list_init_positions, self.list_route_short, self.list_route_long]

        currDir=os.path.dirname(os.path.abspath("__file__"))
        [currDir,er] = currDir.split('Primer-piloto-CoppeliaSim')
        ModelPath = currDir + "Img_Test/"
        self.ModelPath = ModelPath.replace("\\","/")

        if logging.getLogger('').hasHandlers():
            logging.getLogger('').handlers.clear()
        
        logging.basicConfig(
            format = '%(asctime)-5s %(name)-15s %(levelname)-8s %(message)s',
            level  = logging.INFO,      # Nivel de los eventos que se registran en el logger
            filename = "logs_info" + str(date.today()) + ".log", # Fichero en el que se escriben los logs
            filemode = "a"              # a ("append"), en cada escritura, si el archivo de logs ya existe,
                                        # se abre y añaden nuevas lineas.
        )

    def move_agent(self, action):
        if self.routes[self.agent.route][self.agent.position] == (5.2, 5.2, 0):
            if action == 'd':
                self.agent.changeRoute(2)
            elif action == 'a':
                self.agent.changeRoute(1)
        else:
            self.agent.changePosition()

    def make_action(self, action):
        (x,y,theta) = self.routes[self.agent.route][self.agent.position]
        allowed_action = self.dict_posible_outcomes[(x,y,theta)].split(';')
            
        if self.actions[action] in allowed_action:
            Reward_VI=8
            self.move_agent(self.actions[action])
        else:
            Reward_VI=-3
        self.position_Score()
        img = self.get_screen_buffer()
        LR = -0.05
        return Reward_VI+self.TD+LR, img    
        
    def get_screen_buffer(self):
        return np.array(Image.open(self.ModelPath + self.dict_img[self.routes[self.agent.route][self.agent.position]] + '/' + str(np.random.randint(1, BATCH_SIZE)) + '.jpeg').convert('RGB'))

    def step(self,action):           
        reward, img = self.make_action(action)
        is_done = self.is_episode_finished()
        return img,reward,is_done

    def numactions(self):
        return len(self.actions)
    
    def reset(self):
        if (self.reset_moods[self.reset_mood] == 'random'):
            init_position = self.list_posibles_inits[np.random.randint(0, len(self.list_posibles_inits) - 1)]
            if init_position in self.list_init_positions:
                self.agent.reset(self.list_init_positions.index(init_position), 0)
            elif init_position in self.list_route_short:        
                self.agent.reset(self.list_route_short.index(init_position), 1)
            elif init_position in self.list_route_long:        
                self.agent.reset(self.list_route_long.index(init_position), 2)
        else: 
            self.agent.reset(0, 0)   

        logging.info('Initial position: ' + str(self.dict_img[self.routes[self.agent.route][self.agent.position]]) 
            + ' route: ' + str(self.routes[self.agent.route]))    
        
        return self.get_screen_buffer()

    def is_episode_finished(self):
        success = False
        if(self.routes[self.agent.route][self.agent.position] == (2.4, 2.4, 0)):
            success = True        
            logging.info('Is Done!')
        return success

    def position_Score(self):
        self.position = self.routes[self.agent.route][self.agent.position]
        if self.position[0]>-7.5 and self.position[1]>-7.5 and self.position[0]<-6 and self.position[1]<-6:
            self.TD=0
        elif self.position[0]>-7.5 and self.position[1]>-6 and self.position[0]<-6 and self.position[1]<-4.5:
            self.TD=0.05
        elif self.position[0]>-7.5 and self.position[1]>-4.5 and self.position[0]<-6 and self.position[1]<-3:
            self.TD=0.2
        elif self.position[0]>-7.5 and self.position[1]>-3 and self.position[0]<-6 and self.position[1]<-1.5:
            self.TD=0.25
        elif self.position[0]>-6 and self.position[1]>-7.5 and self.position[0]<-4.5 and self.position[1]<-6:
            self.TD=0.15
        elif self.position[0]>-6 and self.position[1]>-6 and self.position[0]<-4.5 and self.position[1]<-4.5:
            self.TD=0.1
        elif self.position[0]>-6 and self.position[1]>-4.5 and self.position[0]<-4.5 and self.position[1]<-3:
            self.TD=0.15
        elif self.position[0]>-6 and self.position[1]>-3 and self.position[0]<-4.5 and self.position[1]<-1.5:
            self.TD=0.3
        elif self.position[0]>-4.5 and self.position[1]>-7.5 and self.position[0]<-3 and self.position[1]<-6:
            self.TD=0.2
        elif self.position[0]>-4.5 and self.position[1]>-6 and self.position[0]<-3 and self.position[1]<-4.5:
            self.TD=0.35
        elif self.position[0]>-4.5 and self.position[1]>-4.5 and self.position[0]<-3 and self.position[1]<-3:
            self.TD=0.4
        elif self.position[0]>-4.5 and self.position[1]>-3 and self.position[0]<-3 and self.position[1]<-1.5:
            self.TD=0.45
        elif self.position[0]>-3 and self.position[1]>-7.5 and self.position[0]<-1.5 and self.position[1]<-6:
            self.TD=0.25
        elif self.position[0]>-3 and self.position[1]>-6 and self.position[0]<-1.5 and self.position[1]<-4.5:
            self.TD=0.3
        elif self.position[0]>-3 and self.position[1]>-4.5 and self.position[0]<-1.5 and self.position[1]<-3:
            self.TD=0.35
        elif self.position[0]>-3 and self.position[1]>-3 and self.position[0]<-1.5 and self.position[1]<-1.5:
            self.TD=0.5
