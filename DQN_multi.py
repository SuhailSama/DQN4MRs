# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 07:41:54 2021

@author: suhai
"""

import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential , load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.


style.use("ggplot")  # setting our style!


LOAD_MODEL = "models\MMR_-9.68avg.model"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 10000 # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 500 # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 500 # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
MODEL_NAME = 'MMR' 
MAX_REWARD = -100  # initial for model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2_000

# Exploration settings
epsilon = 0.99 # not a constant, going to be decayed
EPSILON_DECAY = 0.9976
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 500 # episodes
SHOW_PREVIEW = False


class Blob:
    def __init__(self, size, orient):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)
        self.theta = np.random.randint(0, 360)
        self.vel = np.random.randint(0, 2) # same velocity for all agents
        self.orient = orient

    def __str__(self):
        return f"Blob ({self.x}, {self.y},{self.theta}, {self.vel})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)
    
    def __add__(self, angle): # is it necessary?
        return (self.theta + angle )

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Turn magnet on/off (+45 degrees)
        Change speed: slow, fast
        '''
        if choice == 0: # angle + forward
            self.move(angle = 0, vel = 0)
        elif choice == 1: # angle + NO forward
            self.move(angle = 0, vel = 1.5 )
        elif choice == 2: # No angle + NO forward
            self.move(angle = 1, vel = 0)
        elif choice == 3: # angle + forward
            self.move(angle = 1, vel = 3)

        # print(choice)

    def move(self, vel, angle):
        # If no value for x, move randomly
                
        if self.orient ==1:
            # print(f'confirm p1, old theta = {self.theta}, new theta = {self.theta+angle*43}')
            self.theta += angle*43
        elif self.orient ==2:
            # print(f'confirm p2, old theta = {self.theta}, new theta = {self.theta+ angle*50}')
            self.theta += angle*44
        elif self.orient ==3:
            # print(f'confirm p3, old theta = {self.theta}, new theta = {self.theta+ angle*58}')
            self.theta += angle*45
        elif self.orient ==1:
            # print(f'confirm p1, old theta = {self.theta}, new theta = {self.theta+angle*43}')
            self.theta += angle*46
        elif self.orient ==2:
            # print(f'confirm p2, old theta = {self.theta}, new theta = {self.theta+ angle*50}')
            self.theta += angle*47
        elif self.orient ==3:
            # print(f'confirm p3, old theta = {self.theta}, new theta = {self.theta+ angle*58}')
            self.theta += angle*48
            
            
        self.vel = vel
        x_pos_temp = self.vel *np.cos(math.radians(self.theta)) 
        y_pos_temp = self.vel *np.sin(math.radians(self.theta))
        self.x +=  x_pos_temp.astype(int)
        self.y +=  y_pos_temp.astype(int)
        
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1
#        print(self.theta,vel,self.x,self.y, self.orient)


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 0
    ENEMY_PENALTY = 100
    FOOD_REWARD = 10
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    # OBSERVATION_SPACE_VALUES_DIS = (6,)
    ACTION_SPACE_SIZE = 4
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player1 = Blob(self.SIZE, 1)
        self.player2 = Blob(self.SIZE,2)
        self.player3 = Blob(self.SIZE,3)
        self.player4 = Blob(self.SIZE, 1)
        self.player5 = Blob(self.SIZE,2)
        self.player6 = Blob(self.SIZE,3)
        self.food1 = Blob(self.SIZE,0)
        self.enemy = Blob(self.SIZE,0)
        self.list_players = [self.player1,self.player2,self.player3,self.player4,self.player5,self.player6]
        while self.food1 in self.list_players:
            self.food1 = Blob(self.SIZE,0)
        while self.enemy in self.list_players:
            self.enemy = Blob(self.SIZE,0)
            
        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = np.array(self.player1-self.food1)#[self.player1-self.food1,self.player2-self.food1,self.player3-self.food1]).flatten() #+ (self.player1-self.food1) + (self.player2-self.food1)+ (self.player3-self.food1) 
            # print (f'observation {observation}')
        return observation

    def step(self, action):
        self.episode_step += 1
        # print(f'play1 with action {action}')
        self.player1.action(action)
        # print(f'play2 with action {action}')
        self.player2.action(action)
        # print(f'play3 with action {action}')
        self.player3.action(action)
        # print(f'play1 with action {action}')
        self.player4.action(action)
        # print(f'play2 with action {action}')
        self.player5.action(action)
        # print(f'play3 with action {action}')
        self.player6.action(action)        
        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############
        dist2Food_ind = np.array([self.player1-self.food1,self.player2-self.food1,self.player3-self.food1,self.player4-self.food1,self.player5-self.food1,self.player6-self.food1])
        dist2Food = np.linalg.norm(dist2Food_ind)
        dist2Enemy_ind = np.array([self.player1-self.enemy,self.player2-self.enemy,self.player3-self.enemy,self.player4-self.enemy,self.player5-self.enemy,self.player6-self.enemy])
        dist2Enemy = np.linalg.norm(dist2Enemy_ind)
#        print(dist2Food_ind)
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = dist2Food.flatten()
        done = False
        reward = 0
        for i in range(6): #### number of players
            if self.list_players[i].x <= 0:
                reward -= 1
                # print('hit wall')
            elif self.list_players[i].x >= self.list_players[i].size-1:
                reward -= 1
                # print('hit wall')
            if self.list_players[i].y <= 0:
                reward -= 1
                # print('hit wall')
            elif self.list_players[i].y >= self.list_players[i].size-1:
                reward -= 1
                # print('hit wall')
            
        if self.enemy in self.list_players:
            reward = - self.ENEMY_PENALTY
            done =True
#            print('caught enemy',reward)
        elif self.food1 in self.list_players:
            done =True
            reward =  self.FOOD_REWARD - dist2Food
#            print('one on food',reward)
        else: 
            # print('nothing interesting')
            reward = 0  + dist2Enemy/12
        if self.episode_step >= 300:
            done = True
            # print('episode is over',reward)
        return new_observation, reward, done
    
    def all_the_same(self,elements):
        return len(elements) < 1 or len(elements) == elements.count(elements[0])
    
    def dist(self,blob1,blob2):
        return np.sqrt(np.power(blob1.x-blob2.x,2)+np.power(blob1.y-blob2.y,2))
   
    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
#        cv2.imshow("image", np.array(img))  # show it!
#        cv2.waitKey(1)
        return img
        

    # FOR CNN #
    def get_image(self):
        env = np.ones((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food1.x][self.food1.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player1.x][self.player1.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        env[self.player2.x][self.player2.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        env[self.player3.x][self.player3.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        env[self.player4.x][self.player4.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        env[self.player5.x][self.player5.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        env[self.player6.x][self.player6.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = BlobEnv()

# For stats
ep_rewards = [0]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
sess = tf.Session(config=config) 
backend.set_session(sess)

#
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for  saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


# Agent class
class DQNAgent:
    def __init__(self):

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        if LOAD_MODEL is not None: 
            print(f'loading {LOAD_MODEL}')
            model = load_model(LOAD_MODEL)
            print(f'model {LOAD_MODEL} loaded!')
        else:
            model = Sequential()
    
            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
    
            model.add(Conv2D(256, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
    
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            
            # model.add(Dense(64,input_shape = env.OBSERVATION_SPACE_VALUES_DIS)) # distance
            model.add(Dense(64))
    
            model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
            model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

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
                new_q = (reward + DISCOUNT * max_future_q)
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        # self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
        # return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]



agent = DQNAgent()

# For stats
ep_rewards = []
TEMP_MAX = MAX_REWARD
aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
  
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1
    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    imgs =np.empty([1,300,300,3])
    while not done:
        
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward = reward

        if SHOW_PREVIEW :#and not episode % AGGREGATE_STATS_EVERY:
            img = np.array(env.render()).reshape([1,300,300,3])
            imgs = np.append(imgs,img, axis = 0)
            
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        # print(f'episode reward so far{episode_reward},step number {step}, Done? {done}')
        agent.train(done, step)
        current_state = new_state
        step += 1
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    
    if average_reward > MAX_REWARD:
        print(epsilon)
        # img.append(np.array(env.render()))
        print(f'progress with average reward = {average_reward:0.2f} > {MAX_REWARD}')
        agent.model.save(f'models/{MODEL_NAME}_{average_reward:0.2f}avg.model')
        MAX_REWARD = average_reward
        TEMP_MAX = MAX_REWARD
        if SHOW_PREVIEW:# and not episode % AGGREGATE_STATS_EVERY:
            print('saving video ... ')
            video_name = f'videos/vid{episode}_{average_reward}avg.avi'
            print(video_name)
            height, width, layers = imgs[0].shape
            video = cv2.VideoWriter(video_name, 0, 5,(width,height))
            for image in imgs:
                print(imgs.shape)
                image = np.array(image, dtype=np.uint8)
                video.write(image)
            cv2.destroyAllWindows()
            video.release()
            
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-AGGREGATE_STATS_EVERY:]))
        # Save model, but only when min reward is greater or equal a set value
        
    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
        if epsilon < 0.0011:
            break

AGGREGATE_STATS_EVERY =100
moving_avg = np.convolve(ep_rewards, np.ones((AGGREGATE_STATS_EVERY,))/AGGREGATE_STATS_EVERY, mode='valid')

#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
#plt.scatter([i for i in range(len(ep_rewards))], ep_rewards, label="rewards")
plt.plot([i for i in range(len(moving_avg))], moving_avg, label ="moving average")
plt.legend(loc=4)
plt.ylabel(f"Reward {AGGREGATE_STATS_EVERY} ma")
plt.xlabel("episode #")
plt.show()

# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
