#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:36:20 2020

@author: daniel y erika
"""
import sim                  #V-rep library
import sys
import time                #used to keep track of time
import numpy as np         #array library
import math
import matplotlib as mpl   #used for image plotting
import pandas as pd
import matplotlib.pyplot as plt





class Environment():
    def __init__(self):
        sim.simxFinish(-1) # just in case, close all opened connections

        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)

        if self.clientID!=-1:  #check if client connection successful
            print ('Connected to remote API server')
    
        else:
            print ('Connection not successful')
            sys.exit('Could not connect')
        
        self.EpTime= 0
        self.t1=time.time()
        
        
        self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,'C:/Users/dani-/Documents/Tesis/Mapas Vrep/Robot.ttm',1,sim.simx_opmode_blocking )
        #retrieve pioneer handle
        self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)
        #retrieve motor  handles
        self.errorCode,self.leftmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_oneshot_wait)
        self.errorCode,self.rightmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_oneshot_wait)
    
        #retrieve camera handles
        self.errorCode,self.cameraHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_camera',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.resolution, self.image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_streaming)
        forward = [1,1]
        left = [0,1]
        right = [1,0]
        backward = [-1,-1]
        self.actions = [forward, left, right, backward]
        self.numsteps = 0
        
    
    def make_action(self, action):
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,self.actions[action][0],sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,self.actions[action][1],sim.simx_opmode_oneshot)
    
    def get_screen_buffer(self):
        self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_streaming)
        self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_buffer)
        in_data=np.array(image,dtype=np.uint8)
        in_data.resize([self.resolution[0],self.resolution[1]])
        plt.imshow(in_data,origin='lower')
        return in_data
        
        
    def step(self,action):    

        reward = self.make_action(action)
        img = self.get_screen_buffer()
        is_done=self.is_episode_finished()
        if is_done:
            t2=time.time()
            self.EpTime=t2-self.t1
        img=np.transpose(img)
        
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
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        
        if(self.position[0]>-2.7 and self.position[1]>-2.5 and self.position[0]<-1.9 and self.position[1]<-1.8):
            
            self.returnCode=sim.simxRemoveModel(self.clientID,self.robotHandle,sim.simx_opmode_oneshot_wait)
            self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,'/home/daniel/Documents/Tesis/Mapas Vrep/Robot.ttm',1,sim.simx_opmode_blocking )
            self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
        
            return True
        else:
            return False

        