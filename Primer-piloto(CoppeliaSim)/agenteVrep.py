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
        self.TD=0

        self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,'C:/Users/dani-/Documents/Tesis/Mapas Vrep/Robot.ttm',1,sim.simx_opmode_blocking )
        #retrieve pioneer handle
        self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_streaming)
        #retrieve motor  handles
        self.errorCode,self.leftmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_oneshot_wait)
        self.errorCode,self.rightmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_oneshot_wait)

        self.errorCode,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)

        #retrieve camera handles
        self.errorCode,self.cameraHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_camera',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.resolution, self.image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_streaming)
        
        #retieve Arrows info
        self.Arrows = {}
        a=0
        for i in range(1,5):
            self.errorCode,Arrow = sim.simxGetObjectHandle(self.clientID,'Arrow_L'+str(i),sim.simx_opmode_oneshot_wait)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_streaming)
            time.sleep(0.1)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_buffer)
            self.Arrows["Arrow_L"+str(i)]=[Arrow,Arrow_Pos,a]
            a+=1
        for i in range(1,4):
            self.errorCode,Arrow = sim.simxGetObjectHandle(self.clientID,'Arrow_R'+str(i),sim.simx_opmode_oneshot_wait)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_streaming)
            time.sleep(0.1)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_buffer)
            self.Arrows["Arrow_R"+str(i)]=[Arrow,Arrow_Pos,a]
            a+=1
        print(self.Arrows)

        forward = 'w'
        left = 'a'
        right = 'd'
        backward = 's'
        self.actions = [forward, left, right, backward]
        self.numsteps = 0

    def determine_nearest_Arrow(self,pos_R):
        val=[]
        for i in self.Arrows:
            val.append(abs((self.Arrows[i][1][1]-pos_R[1])/(self.Arrows[i][1][0]-pos_R[0])))
        print('val: ',val)
        comp= min(val)
        keys=list(self.Arrows.keys())
        for i in range(len(val)):
            if comp ==val[i]:
                print(keys[i])
                return keys[i]

    def make_action(self, action):

        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        key_val=self.determine_nearest_Arrow(self.position).split('_')[1]
        Reward_VI=0
        self.move_robot(self.actions[action],2200)
        if key_val[0] == 'L':
            if action == 1:
                Reward_VI=1
            else:
                Reward_VI=0
        elif key_val[0] == 'R':
            if action == 2:
                Reward_VI=1
            else:
                Reward_VI=0

        print(Reward_VI)
        self.position_Score()
        img = self.get_screen_buffer()
        LR= -0.05
        return Reward_VI+self.TD+LR


    def get_screen_buffer(self):
        self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_streaming)
        self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_buffer)
        in_data=np.array(image,dtype=np.uint8)
        in_data.resize([self.resolution[0],self.resolution[1]])
        plt.imshow(in_data,origin='lower')
        return in_data


    def step(self,action):

        reward = self.make_action(action)

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
            self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,'/home/daniel/Documents/Tesis/Mapas Vrep/Robot-2.ttm',1,sim.simx_opmode_blocking )
            self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)

            return True
        else:
            return False

    def rotate(self, orientation):
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        init_g=self.convert_pos_angle(angle[2]*180/np.pi)

        if orientation == 'd':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0.4,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,-0.4,sim.simx_opmode_oneshot)
        elif orientation == 'i':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,-0.4,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0.4,sim.simx_opmode_oneshot)
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        g=self.convert_pos_angle(angle[2]*180/np.pi)

        while abs(abs(init_g)-abs(g)) < 90 or abs(abs(init_g)-abs(g)) > 271:
            err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
            g=self.convert_pos_angle(angle[2]*180/np.pi)
            time.sleep(0.01)

            # print("actual ",g)

        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)

        print("initial gamma ",init_g)
        print("final gamma ",g)

    def convert_pos_angle(self,angle):
        if angle < 0:
            return angle +360
        else:
            return angle

    def move_f_b(self,time_ms, orientation):
        if orientation=='f':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,2.3,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,2.3,sim.simx_opmode_oneshot)
        elif orientation == 'b':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,-2.3,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,-2.3,sim.simx_opmode_oneshot)

        time.sleep(time_ms/1000)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)

    def position_Score(self):
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        if self.position[0]>-7.5 and self.position[1]>-7.5 and self.position[0]<-6 and self.position[1]<-6:
            self.TD=0
        elif self.position[0]>-7.5 and self.position[1]>-6 and self.position[0]<-6 and self.position[1]<-4.5:
            self.TD=0.025
        elif self.position[0]>-7.5 and self.position[1]>-4.5 and self.position[0]<-6 and self.position[1]<-3:
            self.TD=0.1
        elif self.position[0]>-7.5 and self.position[1]>-3 and self.position[0]<-6 and self.position[1]<-1.5:
            self.TD=0.125
        elif self.position[0]>-6 and self.position[1]>-7.5 and self.position[0]<-4.5 and self.position[1]<-6:
            self.TD=0.075
        elif self.position[0]>-6 and self.position[1]>-6 and self.position[0]<-4.5 and self.position[1]<-4.5:
            self.TD=0.05
        elif self.position[0]>-6 and self.position[1]>-4.5 and self.position[0]<-4.5 and self.position[1]<-3:
            self.TD=0.075
        elif self.position[0]>-6 and self.position[1]>-3 and self.position[0]<-4.5 and self.position[1]<-1.5:
            self.TD=0.15
        elif self.position[0]>-4.5 and self.position[1]>-7.5 and self.position[0]<-3 and self.position[1]<-6:
            self.TD=0.1
        elif self.position[0]>-4.5 and self.position[1]>-6 and self.position[0]<-3 and self.position[1]<-4.5:
            self.TD=0.175
        elif self.position[0]>-4.5 and self.position[1]>-4.5 and self.position[0]<-3 and self.position[1]<-3:
            self.TD=0.2
        elif self.position[0]>-4.5 and self.position[1]>-3 and self.position[0]<-3 and self.position[1]<-1.5:
            self.TD=0.225
        elif self.position[0]>-3 and self.position[1]>-7.5 and self.position[0]<-1.5 and self.position[1]<-6:
            self.TD=0.125
        elif self.position[0]>-3 and self.position[1]>-6 and self.position[0]<-1.5 and self.position[1]<-4.5:
            self.TD=0.15
        elif self.position[0]>-3 and self.position[1]>-4.5 and self.position[0]<-1.5 and self.position[1]<-3:
            self.TD=0.175
        elif self.position[0]>-3 and self.position[1]>-3 and self.position[0]<-1.5 and self.position[1]<-1.5:
            self.TD=0.25
            
        

    def move_robot(self, move, time_ms):

        if move == 'w':
            self.move_f_b(time_ms,'f')
        elif move == 'a':
            self.rotate('i')
            self.move_f_b(time_ms,'f')

        elif move == 's':
            self.move_f_b(time_ms,'b')
        elif  move == 'd':
            self.rotate('d')
            self.move_f_b(time_ms,'f')

        else:
            None
