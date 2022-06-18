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
 

MOVE_TIME = 600
WIDTH = 64
HEIGHT = 64
class Environment():
    
    
    def __init__(self):
        sim.simxFinish(-1) # just in case, close all opened connections

        
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,1)

        if self.clientID!=-1:  #check if client connection successful
            print ('Connected to remote API server')

        else:
            print ('Connection not successful')
            sys.exit('Could not connect')

        self.EpTime= 0
        self.t1=time.time()
        self.TD=0
        self.maxintentos = 2   
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
        
        
        currDir=os.path.dirname(os.path.abspath("__file__"))
        print(currDir)
        [currDir,er]=currDir.split('Primer-piloto-CoppeliaSim')
        ModelPath=currDir+"Mapas Vrep"
        self.ModelPath=ModelPath.replace("\\","/")
        
        
        
        self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,self.ModelPath+"/Robot.ttm",1,sim.simx_opmode_blocking )
        pingTime = sim.simxGetPingTime(self.clientID)
        print('Ping time: ', pingTime)
        #print ('Line 40 - code: ', self.returnCode, ' :: basehandle: ', baseHandle)
        #retrieve pioneer handle
        self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_streaming)
        #retrieve motor  handles
        self.errorCode,self.leftmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_oneshot_wait)
        self.errorCode,self.rightmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_oneshot_wait)
        #print ('Line 49 - leftMotor: ', self.leftmotorHandle, ' :: rightMotor: ', self.rightmotorHandle, ':: code: ', self.errorCode)
        self.errorCode,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)

        #retrieve camera handles
        self.errorCode,self.cameraHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_camera',sim.simx_opmode_oneshot_wait)
        #print ('Line 52 - camera: ', self.cameraHandle, ':: code: ', self.errorCode)
        self.returnCode,self.resolution, self.image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_streaming)
        #print ('Line 54 - resolution: ', self.resolution, ' :: img: ', self.image, ':: code: ', self.returnCode)
        
        #retieve Arrows info
        self.Arrows = {}
        a=0
        for i in range(1,6):
            self.errorCode,Arrow = sim.simxGetObjectHandle(self.clientID,'Arrow_L'+str(i),sim.simx_opmode_oneshot_wait)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_streaming)
            time.sleep(0.1)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_buffer)
            self.Arrows["Arrow_L"+str(i)]=[Arrow,Arrow_Pos,a]
            a+=1
        for i in range(1,5):
            self.errorCode,Arrow = sim.simxGetObjectHandle(self.clientID,'Arrow_R'+str(i),sim.simx_opmode_oneshot_wait)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_streaming)
            time.sleep(0.1)
            self.errorCode,Arrow_Pos = sim.simxGetObjectPosition(self.clientID,Arrow,-1,sim.simx_opmode_buffer)
            self.Arrows["Arrow_R"+str(i)]=[Arrow,Arrow_Pos,a]
            a+=1

        forward = 'w'
        left = 'a'
        right = 'd'
        #backward = 's'
        self.actions = [forward, right, left]
        self.numsteps = 0
        
        
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)

    def get_pos_or(self):
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        curr_g=self.convert_pos_angle(angle[2]*180/np.pi)
        self.returnCode,currposition=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)


    def parate(self):
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)
    
    
    ### esta funcion calcula la recompensa con base a la flecha mas cercana, pero se decidio tomar en cuenta solo las posiciones posibles
    def determine_nearest_Arrow(self,pos_R):
        
        val=[]
        for i in self.Arrows:
            val.append(abs((self.Arrows[i][1][1]-pos_R[1])/(self.Arrows[i][1][0]-pos_R[0])))
        comp= min(val)
        keys=list(self.Arrows.keys())
        for i in range(len(val)):
            if comp ==val[i]:
                return keys[i]
            
    def aprox_pos_angle(self,position, angle):
        if  5.95 < abs(position[0]) :
            x= 6.7
        elif  4.5 < abs(position[0]) <= 5.95:
            x= 5.2
        elif  3.1 < abs(position[0]) <= 4.5:
            x= 3.8
        elif abs(position[0]) <= 3.1:
            x= 2.4
            
        if  5.95 < abs(position[1]) :
            y= 6.7
        elif  4.5 < abs(position[1]) <= 5.95:
            y= 5.2
        elif  3.1 < abs(position[1]) <= 4.5:
            y= 3.8
        elif  abs(position[1]) <= 3.1:
            y= 2.4
            
        cur_angle = self.convert_pos_angle(angle[2]*180/np.pi)
        if   45 < cur_angle < 135:
            theta= 90
        elif   135 <= cur_angle < 225:
            theta= 180
        elif   225 <= cur_angle < 315:
            theta= 270
        elif   315 <= cur_angle or  cur_angle <= 45:
            theta= 0  
        return [x,y,theta]


    def make_action(self, action):
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        self.errorCode,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)
         
        (x,y,theta)=self.aprox_pos_angle(self.position, angle)

        allowed_action = self.dict_posible_outcomes[(x,y,theta)].split(';')

            
        if self.actions[action] in allowed_action:
            Reward_VI=3
            self.move_robot(self.actions[action], MOVE_TIME)
        else:
            Reward_VI=-5
        self.position_Score()
        img = self.get_screen_buffer()
        LR = -0.05
        return Reward_VI+self.TD+LR, img
    
        
    def get_screen_buffer(self):
        self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,0,sim.simx_opmode_streaming)
        time.sleep(0.1)
        self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,0,sim.simx_opmode_buffer)
        in_data=np.array(image,dtype=np.uint8)
        time.sleep(0.1)
        in_data.resize([self.resolution[0],self.resolution[1],3])
        in_data = np.flipud( cv2.resize(in_data, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA))
        
        
        return in_data

   

    def step(self,action):   
        try:
            reward, img = self.make_action(action)
            
        except KeyError:
            img= self.get_screen_buffer()
            print('Te perdiste manin, cagaste')
            return(img,-80000,True)

        is_done = self.is_episode_finished()
        if is_done:
            t2=time.time()
            self.EpTime=t2-self.t1
        # img=np.transpose(img)

        return img,reward,is_done

    def numactions(self):
        return len(self.actions)
    
    
    def reset(self):
        self.returnCode=sim.simxRemoveModel(self.clientID,self.robotHandle,sim.simx_opmode_blocking)
        successLoad = True
        for i in range(1,self.maxintentos): 
            try:
                self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,self.ModelPath+"/Robot.ttm",1,sim.simx_opmode_blocking )
                successLoad = True
                break
            except:
                print('no mano, me cague')
                successLoad = False
        if not successLoad :
            sys.exit()
                
        time.sleep(0.1)
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
        
        time.sleep(0.1)
        img = self.get_screen_buffer()


        return img

    def is_episode_finished(self):
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        
        if(2.8 > abs(self.position[0])>1.6 and 2.9 > abs(self.position[1])> 1.6):
            self.returnCode=sim.simxRemoveModel(self.clientID,self.robotHandle,sim.simx_opmode_oneshot_wait)
            self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,self.ModelPath+"/Robot-2.ttm",1,sim.simx_opmode_blocking )
            self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
            print ('\nIs Done!')
            return True
        else:
            return False

    def rotate(self, orientation):
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        init_g=self.convert_pos_angle(angle[2]*180/np.pi)

        if orientation == 'd':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0.1,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,-0.1,sim.simx_opmode_oneshot)
        elif orientation == 'i':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,-0.1,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0.1,sim.simx_opmode_oneshot)
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        g=self.convert_pos_angle(angle[2]*180/np.pi)

        while abs(abs(init_g)-abs(g)) <= 88 or abs(abs(init_g)-abs(g)) >= 268:
            err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
            g=self.convert_pos_angle(angle[2]*180/np.pi)
            

        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)

        #print("initial gamma ",init_g)
        #print("final gamma ",g)
        
        #### prueba de rotacion inmediata con funcion de vrep
    def rotate_orb(self, orientation):
        self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        init_g=self.convert_pos_angle(angle[2]*180/np.pi)
        euler_angle = [0,0,0]
        euler_angle[0] = angle[0]
        euler_angle[1] = angle[1]
        if orientation == 'd':
            euler_angle[2] = (self.convert_pos_angle(init_g -5)*np.pi/180)
            self.errorCode = sim.simxSetObjectOrientation(self.clientID, self.robotHandle, -1,euler_angle,sim.simx_opmode_oneshot)

        elif orientation == 'i':
            euler_angle[2] = (self.convert_pos_angle(init_g + 5)*np.pi/180)
            self.errorCode = sim.simxSetObjectOrientation(self.clientID, self.robotHandle, -1,euler_angle,sim.simx_opmode_oneshot)
            
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
        
    def move_robot(self, move, time_ms):
        if move == 'w':
            self.move_f_b(time_ms,'f')
        elif move == 'a':
            self.rotate('i')
        elif move == 's':
            self.move_f_b(time_ms,'b')
        elif  move == 'd':
            self.rotate('d')
        else:
            None
        
    def move_forward_orb(self):
        move_info = [1.4,0.14] 
        
        err,in_angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        curr_g=self.convert_pos_angle(in_angle[2]*180/np.pi)
        self.returnCode, in_position = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0.2,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0.2,sim.simx_opmode_oneshot)
        end_position = in_position

        if  15 > curr_g or curr_g > 345:
            end_position[0] = in_position[0] + move_info[0]
 
            self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
            while current_pos[0] < (end_position[0] )  :   
               self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
            
        elif 75 < curr_g < 105:    
            end_position[1] = in_position[1] + move_info[0]
           
            self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
            while current_pos[1] < (end_position[1] )  :   
               self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
           
        elif 165 < curr_g < 195:
            end_position[0] = in_position[0] - move_info[0]

            self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
            while current_pos[0] > (end_position[0] )  :   
               self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)

        elif 255 < curr_g < 285:
            end_position[1] = in_position[1] - move_info[0]

            self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
            while current_pos[1] > (end_position[1])  :   
               self.returnCode, current_pos = sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)

        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)