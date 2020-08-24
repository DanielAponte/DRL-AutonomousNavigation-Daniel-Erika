#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from vizdoom import *
import time
import numpy as np
import image_preprocessing

class Environment():
    def __init__(self):
        self.game = DoomGame()
        self.game.load_config("/home/daniel/repos/ViZDoom/scenarios/basic.cfg")
        shoot = [0, 0, 1]
        left = [1, 0, 0]
        right = [0, 1, 0]
        self.actions = [shoot, left, right]
        self.game.init()
        
        
    def step(self,action):        
        reward = self.game.make_action(self.actions[action])
        state = self.game.get_state()
        img = state.screen_buffer
        is_done=self.game.is_episode_finished()
        img=np.transpose(img)
        img= image_preprocessing.convert(img,(80,80))
        return img,reward,is_done
        
    def numactions(self):
        return len(self.actions)
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state()
        img = state.screen_buffer
        img=np.transpose(img)
        img= image_preprocessing.convert(img,(80,80))
        return img

        
