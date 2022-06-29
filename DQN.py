import numpy as np
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as backend

import time
import random
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

from DQNAgent import DQNAgent
from environment import Blob,BlobEnv
# from utilities import ModifiedTensorBoard, movement_animation

style.use("ggplot")  # setting our style!


LOAD_MODEL = None # "models\MMR_5.7avg.model"

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 300 # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 100 # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 50 # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'MMR'
MAX_REWARD = 5.72  # initial for model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 1000

# Exploration settings
epsilon = 0.99 # not a constant, going to be decayed
EPSILON_DECAY = 0.975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50 # episodes
SHOW_PREVIEW = False

env = BlobEnv()

# For stats
ep_rewards = [0]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')



agent = DQNAgent(LOAD_MODEL,MODEL_NAME,env,REPLAY_MEMORY_SIZE,MIN_REPLAY_MEMORY_SIZE,MINIBATCH_SIZE, DISCOUNT, UPDATE_TARGET_EVERY)

# For stats
ep_rewards = []
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
    img =[]
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
        episode_reward += reward


        if SHOW_PREVIEW :#and not episode % AGGREGATE_STATS_EVERY:
             img.append(np.array(env.render()))
                
        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        # print(f'episode reward so far{episode_reward},step number {step}, Done? {done}')
        agent.train(done, step)
        current_state = new_state
        step += 1
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    if average_reward > MAX_REWARD and episode > AGGREGATE_STATS_EVERY:
            # img.append(np.array(env.render()))
            print(f'progress with average reward = {average_reward} > {MAX_REWARD}')
            agent.model.save(f'models/{MODEL_NAME}_{average_reward}avg.model')
            MAX_REWARD = average_reward
            if SHOW_PREVIEW:# and not episode % AGGREGATE_STATS_EVERY:
                print('saving video ... ')
                video_name = f'videos/vid{episode}_{average_reward}avg.avi'
                print(video_name)
                height, width, layers = img[0].shape
                video = cv2.VideoWriter(video_name, 0, 1, (width,height))
                for image in img:
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

moving_avg = np.convolve(ep_rewards, np.ones((AGGREGATE_STATS_EVERY,))/AGGREGATE_STATS_EVERY, mode='valid')

#plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
# plt.scatter([i for i in range(len(ep_rewards))], ep_rewards, label="rewards")
plt.plot([i for i in range(len(moving_avg))], moving_avg, label ="moving average")
plt.legend(loc=4)
plt.ylabel(f"Reward {AGGREGATE_STATS_EVERY} ma")
plt.xlabel("episode #")
plt.show()

# with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
#     pickle.dump(q_table, f)
