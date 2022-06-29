
from keras.models import Sequential , load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam

from collections import deque
from utilities import ModifiedTensorBoard
import time
import random
import numpy as np

# Agent class
class DQNAgent:
    def __init__(self,LOAD_MODEL,MODEL_NAME,env,REPLAY_MEMORY_SIZE,MIN_REPLAY_MEMORY_SIZE,MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY):
        
        self.LOAD_MODEL = LOAD_MODEL
        self.REPLAY_MEMORY_SIZE = REPLAY_MEMORY_SIZE
        self.MIN_REPLAY_MEMORY_SIZE = MIN_REPLAY_MEMORY_SIZE
        self.MODEL_NAME = MODEL_NAME
        self.env = env
        self.MINIBATCH_SIZE = MINIBATCH_SIZE
        self.DISCOUNT = DISCOUNT
        self.UPDATE_TARGET_EVERY = UPDATE_TARGET_EVERY
        # Main model
        
        self.model = self.create_model()
        
        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(self.MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        if self.LOAD_MODEL is not None: 
            print(f'loading {self.LOAD_MODEL}')
            model = load_model(self.LOAD_MODEL)
            print(f'model {self.LOAD_MODEL} loaded!')
        else:
            model = Sequential()
    
            model.add(Conv2D(256, (3, 3), input_shape=self.env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
    
            # model.add(Conv2D(256, (3, 3)))
            # model.add(Activation('relu'))
            # model.add(MaxPooling2D(pool_size=(2, 2)))
            # model.add(Dropout(0.2))
    
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            
            # model.add(Dense(64,input_shape = self.env.OBSERVATION_SPACE_VALUES_DIS)) # distance
            model.add(Dense(64))
    
            model.add(Dense(self.env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        # current_states = np.array([transition[0] for transition in minibatch]) # distance
        
        current_qs_list = self.model.predict(current_states)
        # Get future states from minibatch, then query NN model for Q values
        # When using target netwosteprk, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        # new_current_states = np.array([transition[3] for transition in minibatch]) # distance

        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = (reward + self.DISCOUNT * max_future_q)
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=self.MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        # return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
