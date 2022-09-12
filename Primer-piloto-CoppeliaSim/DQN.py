# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 14:28:14 2021

@author: ErikaMartinezMendez
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
from PIL import Image
from PIL import ImageShow
import numpy as np
import random
import time
import agenteVrep
import agenteVrep_Train
import logging
import msvcrt
from datetime import date
from datetime import datetime

REPLAY_MEMORY_SIZE = 50_000
DISCOUNT = 0.99
MINIBATCH_SIZE = 32
MIN_REPLAY_MEMORY_SIZE = 1_000
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "256x2"

#Number of steps for timeout
TIMEOUT_COUNT = 70

# Exploration settings
epsilon = 0.9  # not a constant, going to be decayed
EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001

# Environment settings
EPISODES = 10_000

env =  agenteVrep.Environment()
# Own Tensorboard class

class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Se eliminan los handlers anteriores
        if logging.getLogger('').hasHandlers():
            logging.getLogger('').handlers.clear()
        
        logging.basicConfig(
            format = '%(asctime)-5s %(name)-15s %(levelname)-8s %(message)s',
            level  = logging.INFO,      # Nivel de los eventos que se registran en el logger
            filename = "logs_info.log", # Fichero en el que se escriben los logs
            filemode = "a"              # a ("append"), en cada escritura, si el archivo de logs ya existe,
                                        # se abre y añaden nuevas lineas.
        )
    def create_model(self):
        env.get_screen_buffer()
        model = Sequential()

        model.add(Conv2D(256, (3, 3), input_shape=(64, 64, 3)))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))        
        
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(1024))
        model.add(Dense(512))
        model.add(Dense(256))
        model.add(Dense(128))
        model.add(Dense(64))
        

        model.add(Dense(len(env.actions), activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
        
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
    
        # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

agent = DQNAgent()

for episode in range(1, EPISODES + 1):

    # Restarting episode - reset episode reward and step number
    print('\nInit episode: ', episode)
    logging.info('Init episode: ' + str(episode))
    actions_analysis = (0,0) #(Random, Q-table)
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    start_t =  time.time()
    
    
    while not done:
        current_t = time.time()
            
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
            actions_analysis = (actions_analysis[0], actions_analysis[1] + 1)
        else:
            # Get random action
            action = np.random.randint(0, env.numactions())
            actions_analysis = (actions_analysis[0] + 1, actions_analysis[1])

        new_state, reward, done = env.step(action)
        

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        im = Image.fromarray(current_state)
        im.save("../Capturas_e_Imagenes/ep"+str(episode)+"img"+str(step)+".jpeg")

        current_state = new_state
        step += 1
        
        if (current_t - start_t ) > 800 :
            done = True
            logging.info('Timeout!')

        if((actions_analysis[0] + actions_analysis[1]) > 100):
            done = True
            logging.info('MaxAttemps!')

            
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    total_actions = actions_analysis[0] + actions_analysis[1]
    print('\nTime dif: ', current_t - start_t )
    print('\nEpsilon: ', epsilon, ' :: Episode_reward ', episode_reward)
    print('\nTotal acts: ', total_actions, ' Random: ', actions_analysis[0],
          ' %% ', actions_analysis[0]/total_actions, 
          ' Q-Table: ', actions_analysis[1],
          ' %% ', actions_analysis[1]/total_actions)
    logging.info('Time dif: ' + str(current_t - start_t) )
    logging.info('Epsilon: ' + str(epsilon) + ' :: Episode_reward ' + str(episode_reward))
    logging.info('Total acts: ' + str(total_actions) + ' Random: ' + str(actions_analysis[0]) +
          ' %% ' + str(actions_analysis[0]/total_actions) + 
          ' Q-Table: ' + str(actions_analysis[1]) +
          ' %% ' + str(actions_analysis[1]/total_actions))
    
    if msvcrt.kbhit():
        if msvcrt.getch() == b'\x1b':
            break

print('Saving Model...')
agent.model.save('model' + str(date.today()) + '.model')
agent.target_model.save('target_model' + str(date.today()) + '.model')
        
        
        
        