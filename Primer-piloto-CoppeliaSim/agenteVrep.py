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
import environment_params as env

SIMULATION_TYPE = "Train"
MAP = "6x6"
MOVE_TIME = 625
WIDTH = 64
HEIGHT = 64
MOVE_DISTANCE = 1.5

class Environment():
    def __init__(self):
        ### Variables definition
        self.params_env = env.ParametersEnvironments(SIMULATION_TYPE + "_" + MAP)
        self.move_directions = {
            0: (1, 0, 0),
            90: (0, 1, 1),
            180: (-1, 0, 0),
            270: (0, -1, 1)
        }
        self.max_attemps = 5
        self.TD = 0

        forward = 'w'
        left = 'a'
        right = 'd'
        #backward = 's'
        self.actions = [left, right, forward]

        sim.simxFinish(-1) # just in case, close all opened connections   
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,1)

        if self.clientID!=-1:  #check if client connection successful
            print ('Connected to remote API server')
        else:
            print ('Connection not successful')
            sys.exit('Could not connect')

        currDir = os.path.dirname(os.path.abspath("__file__"))
        [currDir, er] = currDir.split('Primer-piloto-CoppeliaSim')
        ModelPath = currDir + "Mapas Vrep"
        self.ModelPath = ModelPath.replace("\\", "/")

        self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,self.ModelPath+"/Robot-5.ttm",1,sim.simx_opmode_blocking )
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

        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)
        
        # Se eliminan los handlers anteriores
        if logging.getLogger('').hasHandlers():
            logging.getLogger('').handlers.clear()
        
        logging.basicConfig(
            format = '%(asctime)-5s %(name)-15s %(levelname)-8s %(message)s',
            level  = logging.INFO,      # Nivel de los eventos que se registran en el logger
            filename = "logs_info" + str(date.today()) + ".log", # Fichero en el que se escriben los logs
            filemode = "a"              # a ("append"), en cada escritura, si el archivo de logs ya existe,
                                        # se abre y añaden nuevas lineas.
        )
            
    def aprox_pos_angle_v2(self, angle):
        def aprox_pos(position):
            for p in self.params_env.positions_list:
                if(p+1) > position > (p-1):
                    return p
        x = aprox_pos(self.position[0])
        y = aprox_pos(self.position[1])

        cur_angle = self.convert_pos_angle(angle[2]*180/np.pi)
        if   45 < cur_angle < 135:
            theta= 90
        elif   135 <= cur_angle < 225:
            theta= 180
        elif   225 <= cur_angle < 315:
            theta= 270
        elif   315 <= cur_angle or  cur_angle <= 45:
            theta= 0  
        return [x, y, theta]

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
        is_lost = True      #Variable utilizada para definir recompensa
        is_valid_action = True      #Variable utilizada en sistema de protección
        is_correct_action = False

        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        self.errorCode,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)
         
        self.position = (self.position[0] + 6.7, self.position[1] + 6.7)
        (x,y,theta)=self.aprox_pos_angle_v2(angle)

        is_valid_action = (x,y,theta) in self.params_env.list_allowed_forward_actions if self.actions[action] == 'w' else True

        if SIMULATION_TYPE == "Train": 
            try:
                allowed_action = self.params_env.dict_posible_outcomes[(x, y, theta)].split(';')
                is_lost = False
            except:
                allowed_action = [
                    self.actions[2]] if is_valid_action and self.actions[action] == 'w' else [self.actions[0]]

            if is_lost and self.actions[action] in allowed_action:
                is_correct_action = True
                Reward_VI = 0.3
            elif self.actions[action] in allowed_action:
                is_correct_action = True
                Reward_VI = 0.9
            else:
                Reward_VI = -0.8        
            self.position_score_v2()
        else: 
            Reward_VI = 0

        self.move_robot(self.actions[action], (x, y, theta)) if is_valid_action else None

        img = self.get_screen_buffer()
        LR = -0.2
        return Reward_VI + self.TD + LR, img, is_correct_action
        
    def get_screen_buffer(self):
        try:
            self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,0,sim.simx_opmode_streaming)
            time.sleep(0.3)
            self.returnCode,self.resolution, image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,0,sim.simx_opmode_buffer)
            in_data=np.array(image,dtype=np.uint8)
            time.sleep(0.3)
            in_data.resize([self.resolution[0],self.resolution[1],3])
            in_data = np.flipud( cv2.resize(in_data, (WIDTH,HEIGHT), interpolation = cv2.INTER_AREA))
        except:
            print('\nError get screen buffer')
            logging.info('Error get screen buffer, in_data: ' +  str(in_data) + ' image: ' + str(image))
        return in_data   

    def step(self, action):
        reward, img, is_correct_action = self.make_action(action)
        is_done = self.is_episode_finished()
        return img, reward, is_done, is_correct_action

    def numactions(self):
        return len(self.actions)
    
    def reset(self):
        self.returnCode=sim.simxRemoveModel(self.clientID,self.robotHandle,sim.simx_opmode_blocking)
        successLoad = True
        for i in range(1,self.max_attemps): 
            try:
                agent_model = self.params_env.get_agent_model()
                self.returnCode, baseHandle = sim.simxLoadModel(self.clientID, self.ModelPath + agent_model, 1, sim.simx_opmode_blocking)
                successLoad = True
                logging.info('Se configuró correctamente el modelo')
                break
            except:
                print('Imposible configurar el modelo')
                logging.info('Imposible configurar el modelo')
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
        for goal in self.params_env.goal_object_position_list:
            if (goal[0]+1) > self.position[0] > (goal[0]-1) and (goal[1]+1) > self.position[1] > (goal[1]-1):
                logging.info('Is Done!')
                return True
        return False

    def rotate(self, direction):
        error, current_orientation = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_buffer)
        current_angle = self.get_current_angle(current_orientation[2])
        target_angle, direction_motors = self.get_target_angle(current_angle, direction)

        while not(0.3 > np.abs(current_angle-target_angle) > 0):
            error, current_orientation = sim.simxGetObjectOrientation(self.clientID, self.robotHandle, -1, sim.simx_opmode_buffer)
            current_angle = self.convert_pos_angle(current_orientation[2]*180/np.pi)
            velocity = self.get_velocity(current_angle, target_angle, 90, 2)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.leftmotorHandle, velocity*direction_motors[0], sim.simx_opmode_oneshot)            
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.rightmotorHandle, velocity*direction_motors[1], sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(
                    self.clientID, self.rightmotorHandle, 0, sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(
                    self.clientID, self.leftmotorHandle, 0, sim.simx_opmode_oneshot)

    def get_velocity(self, current, target, normalizer, multiplier):
        dif = np.abs(target - current)
        if(dif > normalizer):
            dif = normalizer
        return (dif/normalizer)*multiplier

    def get_target_angle(self, current_angle, direction):
        if direction == self.actions[0]:
            target_angle = current_angle + 90
            direction_motors = [-1, 1]
        else:
            target_angle = current_angle - 90
            direction_motors = [1, -1]

        if (direction == self.actions[0]) and (current_angle == 360):
            target_angle = 90
        elif (direction == self.actions[1]) and (current_angle == 0):
            target_angle = 270
        return target_angle, direction_motors

    def get_current_angle(self, current_orientation):
        current_angle = self.convert_pos_angle(current_orientation*180/np.pi)
        if  315 >= current_angle > 225:
            current_angle = 270
        elif 225 >= current_angle > 135:
            current_angle = 180
        elif 135 >= current_angle > 45:
            current_angle = 90
        elif current_angle > 315:
            current_angle = 360
        elif current_angle <= 45:
            current_angle = 0
        return current_angle

    def convert_neg_angle(self,angle):
        if angle > 180:
            return angle -360
        else:
            return angle
    
    def convert_pos_angle(self,angle):
        if angle < 0:
            return angle + 360
        elif angle > 360:
            return angle - 360
        else:
            return angle

    def move_f_b(self, orientation, XYT):
        current = (XYT[0], XYT[1])
        move_direction = self.move_directions[XYT[2]]
        target = (move_direction[0] * MOVE_DISTANCE, move_direction[1] * MOVE_DISTANCE)
        target = (target[0] + current[0], target[1] + current[1])
        while not (0.03 > np.abs(target[move_direction[2]] - current[move_direction[2]]) > 0):
            self.returnCode, self.position = sim.simxGetObjectPosition(self.clientID, self.robotHandle, sim.sim_handle_parent, sim.simx_opmode_buffer)
            current = (self.position[0] + 6.7, self.position[1] + 6.7)
            velocity = (np.abs(target[move_direction[2]] - current[move_direction[2]])/MOVE_DISTANCE) * 3
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.leftmotorHandle, velocity, sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.rightmotorHandle, velocity, sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.leftmotorHandle, 0, sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID, self.rightmotorHandle, 0, sim.simx_opmode_oneshot)

    def position_score(self):
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,sim.sim_handle_parent,sim.simx_opmode_buffer)
        if self.position[0]>-7.5 and self.position[1]>-7.5 and self.position[0]<-6 and self.position[1]<-6:
            self.TD=0
        elif self.position[0]>-7.5 and self.position[1]>-6 and self.position[0]<-6 and self.position[1]<-4.5:
            self.TD=0.03
        elif self.position[0]>-7.5 and self.position[1]>-4.5 and self.position[0]<-6 and self.position[1]<-3:
            self.TD=0.12
        elif self.position[0]>-7.5 and self.position[1]>-3 and self.position[0]<-6 and self.position[1]<-1.5:
            self.TD=0.15
        elif self.position[0]>-6 and self.position[1]>-7.5 and self.position[0]<-4.5 and self.position[1]<-6:
            self.TD=0.09
        elif self.position[0]>-6 and self.position[1]>-6 and self.position[0]<-4.5 and self.position[1]<-4.5:
            self.TD=0.06
        elif self.position[0]>-6 and self.position[1]>-4.5 and self.position[0]<-4.5 and self.position[1]<-3:
            self.TD=0.09
        elif self.position[0]>-6 and self.position[1]>-3 and self.position[0]<-4.5 and self.position[1]<-1.5:
            self.TD=0.18
        elif self.position[0]>-4.5 and self.position[1]>-7.5 and self.position[0]<-3 and self.position[1]<-6:
            self.TD=0.12
        elif self.position[0]>-4.5 and self.position[1]>-6 and self.position[0]<-3 and self.position[1]<-4.5:
            self.TD=0
        elif self.position[0]>-4.5 and self.position[1]>-4.5 and self.position[0]<-3 and self.position[1]<-3:
            self.TD=0.24
        elif self.position[0]>-4.5 and self.position[1]>-3 and self.position[0]<-3 and self.position[1]<-1.5:
            self.TD=0.27
        elif self.position[0]>-3 and self.position[1]>-7.5 and self.position[0]<-1.5 and self.position[1]<-6:
            self.TD=0.15
        elif self.position[0]>-3 and self.position[1]>-6 and self.position[0]<-1.5 and self.position[1]<-4.5:
            self.TD=0.18
        elif self.position[0]>-3 and self.position[1]>-4.5 and self.position[0]<-1.5 and self.position[1]<-3:
            self.TD=0.21
        elif self.position[0]>-3 and self.position[1]>-3 and self.position[0]<-1.5 and self.position[1]<-1.5:
            self.TD=0.3
    
    def position_score_v2(self):
        for position in self.params_env.close2objective_list:
            if (position[0]+1) > self.position[0] and (position[1]+1) > self.position[1] and (position[0]-1) < self.position[0] and (position[1]-1) < self.position[1]:
                self.TD = position[2]
                break
            
        
    def move_robot(self, move, XYT):
        if move == 'w':
            self.move_f_b('f', XYT)
        elif move == 'a':
            self.rotate('a')
        elif move == 's':
            self.move_f_b('b', XYT)
        elif move == 'd':
            self.rotate('d')
        else:
            None

# env = Environment()
# env.reset()
# while(True):
#     tecla = input()
#     if tecla== 'w':
#         env.step(2)
#     elif tecla== 'a':
#         env.step(0)
#     elif tecla== 'd':
#         env.step(1)
#     else: 
#         env.reset()