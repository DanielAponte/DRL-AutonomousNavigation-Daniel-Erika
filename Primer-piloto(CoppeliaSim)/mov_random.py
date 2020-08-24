#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:27:30 2020

@author: daniel
"""
import agenteVrep
import random


env=agenteVrep.Environment()
while True:
    env.step(random.randint(0, 3))
