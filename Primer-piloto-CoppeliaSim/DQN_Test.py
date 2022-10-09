# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:28:14 2021

@author: ErikaMartinezMendez
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
from PIL import Image
from PIL import ImageShow
import numpy as np
import random
import time
import agenteVrep
import agenteVrep_Train
import logging
from datetime import date
from datetime import datetime
import tensorflow as tf
import json
import os

MODEL_NAME = "model2022-10-08.model"
MODEL_NAME_SAVE = "ModelDQN_Test"         
TARGET_MODEL_NAME = "target_model2022-10-08.model"  
REPLAY_MEMORY_NAME = "replay_memory2022-10-08.json" 

TIMEOUT_MAX = 100
AGGREGATE_STATS_EVERY = 1

#Number of steps for timeout
TIMEOUT_COUNT = 70

# Environment settings
EPISODES = 12

env =  agenteVrep.Environment()

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.step +=1
                self.writer.flush()
            
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DQNAgent:
    def __init__(self):
        self.model, self.target_model, self.replay_memory = self.load_agent()

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

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

    def load_agent(self):
        model = self.load_model(MODEL_NAME)

        # Sección de código para lectura de modelo pre-calentado
        # target_model = model
        # replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Seccion de código para lectura de modelo entrenado
        target_model = self.load_model(TARGET_MODEL_NAME)

        file_json = open(REPLAY_MEMORY_NAME)
        replay_memory = deque(json.load(file_json))
        file_json.close()        

        return model, target_model, replay_memory

    def load_model(self, model):
        return tf.keras.models.load_model(model)

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

agent = DQNAgent()   
ep_reward = []
for episode in range(1, EPISODES + 1):

    # Restarting episode - reset episode reward and step number
    print('\nInit episode: ', episode)
    logging.info('Init episode: ' + str(episode))
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    finished = False
    start_t =  time.time()
    while not (done or finished):
        current_t = time.time()

        action = np.argmax(agent.get_qs(current_state))   
        new_state, reward, done, error = env.step(action)

        if (current_t - start_t ) > TIMEOUT_MAX :
            finished = True
            reward = -1
            logging.info('Timeout!')

        if(step > 50):
            finished = True
            reward = -1
            logging.info('MaxAttemps!')

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        current_state = new_state
        step += 1
        ep_reward.append(episode_reward)
 
    print('\nTime dif: ', current_t - start_t )
    print('\nEpisode_reward: ', episode_reward)
    print('\nTotal acts: ', step)
    logging.info('Time dif: ' + str(current_t - start_t))
    logging.info('Episode_reward ' + str(episode_reward))
    logging.info('Total acts: ' + str(step))
