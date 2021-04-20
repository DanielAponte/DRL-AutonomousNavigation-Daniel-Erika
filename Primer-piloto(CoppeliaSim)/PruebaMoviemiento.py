# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:25:38 2020

@author: daniel y erika 


este script tiene como  objtivo mover el robot y otras pruebitas

"""

import sim
from msvcrt import getch
import time
import numpy as np


class Environment():
    def __init__(self):
        sim.simxFinish(-1) # just in case, close all opened connections
        self.clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)
        if self.clientID!=-1:
            print ('Connected to remote API server')
        else:             
            print ('Connection not successful')
        #self.returnCode,baseHandle=sim.simxLoadModel(self.clientID,'C:/Users/dani-/Documents/Tesis/Mapas Vrep/Robot.ttm',1,sim.simx_opmode_blocking )
        #retrieve pioneer handle
        self.errorCode,self.robotHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.position=sim.simxGetObjectPosition(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)
        #retrieve motor  handles
        
        self.errorCode,self.rightmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_oneshot_wait)
        self.errorCode,self.leftmotorHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_oneshot_wait)
    
        #retrieve camera handles
        self.errorCode,self.cameraHandle=sim.simxGetObjectHandle(self.clientID,'Pioneer_camera',sim.simx_opmode_oneshot_wait)
        self.returnCode,self.resolution, self.image=sim.simxGetVisionSensorImage( self.clientID,self.cameraHandle,1,sim.simx_opmode_streaming)
        forward = [1,1]
        left = [0,1]
        right = [1,0]
        backward = [-1,-1]
        self.actions = [forward, left, right, backward]
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_streaming)
    def girar(self, sentido):
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)
        init_g=self.convert_pos_angle(angle[2]*180/np.pi)
        
        if sentido == 'd':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0.4,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,-0.4,sim.simx_opmode_oneshot)
        elif sentido == 'i':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,-0.4,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0.4,sim.simx_opmode_oneshot)  
        err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)           
        g=self.convert_pos_angle(angle[2]*180/np.pi)
        
        while abs(abs(init_g)-abs(g)) < 90 or abs(abs(init_g)-abs(g)) > 271:
            err,angle=sim.simxGetObjectOrientation(self.clientID,self.robotHandle,-1,sim.simx_opmode_buffer)           
            g=self.convert_pos_angle(angle[2]*180/np.pi)
            time.sleep(0.01)
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
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,3,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,3,sim.simx_opmode_oneshot)
        elif orientation == 'b':
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,-3,sim.simx_opmode_oneshot)
            self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,-3,sim.simx_opmode_oneshot)
        
        time.sleep(time_ms/1000)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.rightmotorHandle,0,sim.simx_opmode_oneshot)
        self.errorCode = sim.simxSetJointTargetVelocity(self.clientID,self.leftmotorHandle,0,sim.simx_opmode_oneshot)
        
        
        
    def mover_robot(self, tecla, time_ms):
        
        if tecla== 'w':
            self.move_f_b(time_ms,'f')
        elif tecla== 'a':
            self.girar('i')
            self.move_f_b(time_ms,'f')
            
        elif tecla== 's':
            self.move_f_b(time_ms,'b')
        elif tecla== 'd':
            self.girar('d')
            self.move_f_b(time_ms,'f')
            
        else:
            None
        