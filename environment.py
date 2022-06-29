import numpy as np
import cv2
from PIL import Image
import math


class BlobEnv:
    SIZE = 5
    RETURN_IMAGES = True
    MOVE_PENALTY = 0
    ENEMY_PENALTY = 10
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
        # self.player2 = Blob(self.SIZE,2)
        # self.player3 = Blob(self.SIZE,3)
        self.food1 = Blob(self.SIZE,0)
        self.enemy = Blob(self.SIZE,0)
        self.list_players = [self.player1]#,self.player2,self.player3]
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
        # self.player2.action(action)
        # print(f'play3 with action {action}')
        # self.player3.action(action)
        
        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = np.array(self.player1-self.food1)#[self.player1-self.food1,self.player2-self.food1,self.player3-self.food1]).flatten()
        done = False
        reward = 0
        for i in range(1): #### number of players
            if self.list_players[i].x <= 0:
                reward -= 1
            elif self.list_players[i].x >= self.list_players[i].size-1:
                reward -= 1
            if self.list_players[i].y <= 0:
                reward -= 1
            elif self.list_players[i].y >= self.list_players[i].size-1:
                reward -= 1
            
        if self.enemy in self.list_players:
            reward = - self.ENEMY_PENALTY
            done =True
        elif self.food1 in self.list_players:
            done =True
            reward =  self.FOOD_REWARD - np.sum(self.dist(self.player1,self.food1))#[self.dist(self.player1,self.food1),self.dist(self.player2,self.food1),self.dist(self.player3,self.food1)])/3
        else: 
            reward = 0 
                     # + np.amin([self.dist(self.player1,self.enemy),self.dist(self.player2,self.enemy),self.dist(self.player3,self.enemy)])
        if self.episode_step >= 200:
            done = True
        return new_observation, reward, done
    
    def all_the_same(self,elements):
        return len(elements) < 1 or len(elements) == elements.count(elements[0])
    
    def dist(self,blob1,blob2):
        return np.sqrt(np.power(blob1.x-blob2.x,2)+np.power(blob1.y-blob2.y,2))
   
    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)
        return img
        

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food1.x][self.food1.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player1.x][self.player1.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        # env[self.player2.x][self.player2.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        # env[self.player3.x][self.player3.y] = self.d[self.PLAYER_N]  # sets the player tile to blue

        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img




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
            self.theta += angle*45
        elif self.orient ==3:
            # print(f'confirm p3, old theta = {self.theta}, new theta = {self.theta+ angle*58}')
            self.theta += angle*48
        self.vel = vel
        x_pos_temp = self.vel *np.cos(math.radians(self.theta)) 
        y_pos_temp = self.vel *np.sin(math.radians(self.theta))
        self.x +=  x_pos_temp.astype(int)
        self.y +=  y_pos_temp.astype(int)
  
            # print('no action')
        
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1
        # print(self.theta,vel,self.x,self.y, self.orient)

