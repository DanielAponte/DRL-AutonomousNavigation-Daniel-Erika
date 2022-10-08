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
import msvcrt
from datetime import date
from datetime import datetime
import tensorflow as tf
import json
import os

AGENT_INIT = "LOAD"   # OPTIONS: CREATE, LOAD
MODEL_NAME = "model2022-10-06.model"
MODEL_NAME_SAVE = "ModelDQN_Entrenamiento"         # Necessary when AGENT_INIT = "LOAD"
TARGET_MODEL_NAME = "target_model2022-10-06.model"  # Necessary when AGENT_INIT = "LOAD"
REPLAY_MEMORY_NAME = "replay_memory2022-10-06.json" # Necessary when AGENT_INIT = "LOAD"

REPLAY_MEMORY_SIZE = 50_000
DISCOUNT = 0.99
MINIBATCH_SIZE = 64
MIN_REPLAY_MEMORY_SIZE = 0
UPDATE_TARGET_EVERY = 5
TIMEOUT_MAX = 300
AGGREGATE_STATS_EVERY = 1
VALIDATION_LEARNING_POLICY = 20
CHANGE_RESET_EVERY = 10

#Number of steps for timeout
TIMEOUT_COUNT = 70

# Exploration settings
epsilon = 0.5  # not a constant, going to be decayed
epsilon_tmp = 0.5
EPSILON_DECAY = 0.999
MIN_EPSILON = 0
MAX_EPSILON = 0.5

# Environment settings
EPISODES = 200

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

        if(AGENT_INIT == "CREATE"):
            self.model, self.target_model, self.replay_memory = self.create_agent()
        else:
            self.model, self.target_model, self.replay_memory = self.load_agent()

        self.tensorboard = ModifiedTensorBoard(log_dir="tb_logs/{}-{}".format(MODEL_NAME_SAVE, int(time.time())))

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

    def create_agent(self): 
        model = self.create_model()
        target_model = self.create_model()
        target_model.set_weights(model.get_weights())
        replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        return model, target_model, replay_memory

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
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        

        model.add(Dense(len(env.actions), activation=Activation('softmax')))  # Métodos de activación disp. sigmoid o mejor softmax
        model.compile(loss="mse", optimizer=Adam(), metrics=['accuracy'])
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
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=2, shuffle=False, 
                callbacks = [self.tensorboard] if terminal_state else None)    

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def save_model(self): 
        print('Saving model...')
        agent.model.save('model' + str(date.today()) + '.model')
        agent.target_model.save('target_model' + str(date.today()) + '.model')
        print('Model saved')

        json_object = json.dumps(list(self.replay_memory), cls = NpEncoder)
        with open('replay_memory' + str(date.today()) + '.json', 'w') as f:
            f.write(json_object)
        logging.info('Model and replay memory saved' + str(date.today()))

    def define_learning_policy(self, episode, epsilon):
        # epsilon = self.get_decay_epsilon(epsilon, episode)
        if episode % VALIDATION_LEARNING_POLICY == 0 or episode == 1:
            epsilon = 0
        else:
            # epsilon = 0.5
            epsilon = self.get_decay_epsilon(epsilon, episode)
            epsilon_tmp = epsilon
        # if episode % CHANGE_RESET_EVERY:
        #     reset_mood_bool = not bool(env.reset_mood)
        #     env.reset_mood = int(reset_mood_bool)             
        return epsilon
    
    def get_decay_epsilon(self, epsilon, episode):
        # Decay epsilon
        # if epsilon > MIN_EPSILON:
        #     epsilon *= EPSILON_DECAY
        #     epsilon = max(MIN_EPSILON, epsilon)
        if epsilon_tmp > MIN_EPSILON:
            epsilon = ((195-episode)/195)*MAX_EPSILON
        return epsilon

agent = DQNAgent()   
reward_hist = []
epsilon_hist = []
total_acts_hist = []
total_error_acts_hist = []
for episode in range(1, EPISODES + 1):

    # Restarting episode - reset episode reward and step number
    print('\nInit episode: ', episode)
    logging.info('Init episode: ' + str(episode))
    actions_analysis = (0,0) #(Random, Q-table)
    episode_reward = 0
    step = 1
    error_acts = 0  

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    finished = False
    start_t =  time.time()
    epsilon = agent.define_learning_policy(episode, epsilon)
    while not (done or finished):
        current_t = time.time()
            
        # This part stays mostly the same, the change is to query a model for Q values
        is_predict_action = False
        if np.random.random() > epsilon:
            # Get action from Q table
            is_predict_action = True
            action = np.argmax(agent.get_qs(current_state))
            actions_analysis = (actions_analysis[0], actions_analysis[1] + 1)
        else:
            # Get random action
            action = np.random.randint(0, env.numactions())
            actions_analysis = (actions_analysis[0] + 1, actions_analysis[1])

        new_state, reward, done, is_correct_action = env.step(action)
        
        if not(is_predict_action and is_correct_action):
            error_acts += 1

        if (current_t - start_t ) > TIMEOUT_MAX :
            finished = True
            reward = -1
            logging.info('Timeout!')

        if((actions_analysis[0] + actions_analysis[1]) > 75):
            finished = True
            reward = -1
            logging.info('MaxAttemps!')

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        # im = Image.fromarray(current_state)
        # im.save("../Capturas_e_Imagenes/ep"+str(episode)+"img"+str(step)+".jpeg")

        current_state = new_state
        step += 1

    total_actions = actions_analysis[0] + actions_analysis[1]        
    reward_hist.append(episode_reward)
    epsilon_hist.append(epsilon)
    total_acts_hist.append(total_actions)
    total_error_acts_hist.append(error_acts)    

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

agent.save_model()
print('\n Reward_hist: ', reward_hist)
print('\n Epsilon_hist: ', epsilon_hist)
print('\n Total_acts_hist: ', total_acts_hist)
print('\n Total_error_acts_hist: ', total_error_acts_hist)

logging.info('Reward_hist' + str(reward_hist) )
logging.info('Epsilon_hist' + str(epsilon_hist) )
logging.info('Total_acts_hist' + str(total_acts_hist) )
logging.info('Total_error_acts_hist' + str(total_error_acts_hist) )
        
        