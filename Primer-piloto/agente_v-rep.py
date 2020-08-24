#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:36:20 2020

@author: daniel y erika
"""
import vrep                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting
import pandas as pd
import matplotlib.pyplot as plt





class Environment():
    def __init__(self):
        vrep.simxFinish(-1) # just in case, close all opened connections

        self.clientID=vrep.simxStart('127.0.0.1',19999,True,True,5000,5)

        if self.clientID!=-1:  #check if client connection successful
            print ('Connected to remote API server')
    
        else:
            print ('Connection not successful')
            sys.exit('Could not connect')
        
        #retrieve pioneer handle
        self.errorCode,self.robotHandle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
        self.returnCode,self.position=vrep.simxGetObjectPosition(self.clientID,self.robotHandle,-1,vrep.simx_opmode_streaming)
        #retrieve motor  handles
        self.errorCode,self.leftmotorHandle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',vrep.simx_opmode_oneshot_wait)
        self.errorCode,self.rightmotorHandle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',vrep.simx_opmode_oneshot_wait)
    
        #retrieve camera handles
        self.errorCode,self.cameraHandle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_camera',vrep.simx_opmode_oneshot_wait)
        self.returnCode,self.resolution, self.image=vrep.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,vrep.simx_opmode_streaming)
        forward = [5,5]
        left = [0,5]
        right = [5,0]
        backward = [-5,-5]
        self.actions = [forward, left, right, backward]
        
    
    def make_action(self, action):
        self.errorCode = vrep.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,self.actions[action][0],vrep.simx_opmode_oneshot)
        self.errorCode = vrep.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,self.actions[action][1],vrep.simx_opmode_oneshot)
    
    def get_screen_buffer(self):
        self.returnCode,self.resolution, image=vrep.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,vrep.simx_opmode_streaming)
        self.returnCode,self.resolution, image=vrep.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,vrep.simx_opmode_buffer)
        in_data=np.array(image,dtype=np.uint8)
        in_data.resize([self.resolution[0],self.resolution[1]])
        plt.imshow(in_data,origin='lower')
        return in_data
        
        
    def step(self,action):        
        reward = self.make_action(action)
        img = self.get_screen_buffer()
        is_done=self.is_episode_finished()
        img=np.transpose(img)
        img= image_preprocessing.convert(img,(80,80))
        return img,reward,is_done
        
    def numactions(self):
        return len(self.actions)
    def reset(self):
        self.new_episode()
        state = self.game.get_state()
        img = state.screen_buffer
        img=np.transpose(img)
        img= image_preprocessing.convert(img,(80,80))
        return img
    
    def is_episode_finished(self):
        self.returnCode,self.position=vrep.simxGetObjectPosition(self.clientID,self.robotHandle,-1,vrep.simx_opmode_buffer)
        
        if(self.position[0]>-2.7 and self.position[1]>-2.5 and self.position[0]<-1.9 and self.position[1]<-1.8):
            vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
            self.returnCode=vrep.simxRemoveObject(self.clientID,self.robotHandle,vrep.simx_opmode_oneshot_wait)
            self.errorCode,self.robotHandle=vrep.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',vrep.simx_opmode_oneshot_wait)
            return True
        else:
            return False
    
