#Importing the libraries needed 
import os, sys
import gym
import time
import matplotlib.pyplot as plt 
import numpy as np

import text_flappy_bird_gym
import json
import pandas as pd

#Q-table saving function
# def save_json(dictionary):
#     str_keys = str(dictionary.keys())
#     dicti
#     with open(SAVE_PATH+'qtable.json','w') as fp:
#         json.dump(dictionary,fp)

class QLearning:

    def __init__(self,environment,alpha,gamma,epsilon):

        self.actions = environment.action_space.n
        self.alpha = alpha 
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.Q_table = {} #Key: Coordinate position (State), Value: Action


    #Argmax function
    def argmax(self,q_values):
        
        top = float('-inf')
        ties = []
        
        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []
                
            if q_values[i] == top:
                ties.append(i)
                
        return np.random.choice(ties)

    #Custom Reward function 
    def give_reward(self,done_state):
        if not done_state:
            return 1
        else:
            return -1000
    
    #Helper function for initializing Q-values
    def initialize_q_values(self,state):
        #Check if the state exists in the table 
        #If not, then load in the initialized value as 0
        self.Q_table[state] = [0,0]
    
    #Update Q-table with the new values
    def update_q_values(self,state,action_pair):
        self.Q_table[state] = action_pair

    def agent_initialize(self,state):
        current_q = self.Q_table[state]
        #Choose a random action using epsilon-greedy strategy 
        random_action_prob = np.random.random()
        if random_action_prob < self.epsilon:
            #Choose a random action to explore 
            action = np.random.randint(self.actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action #Return the action that maximizes the Q-value 
    
    def agent_step(self,reward,state):
        # current_q = self.Q_table[state]
        # random_action_prob = np.random.random()
        # if random_action_prob < self.epsilon:
        #     action = np.random.randint(self.actions)
        # else:
        #     action = self.argmax(current_q)
            
        #Q-update equation
        target_update = reward + self.gamma*np.max(self.Q_table[state][0:2])
        Q_prev = self.Q_table[self.prev_state][self.prev_action] #Take the Q-value from the 
        Q_prev += self.alpha*(target_update - Q_prev)
        self.Q_table[self.prev_state][self.prev_action] = Q_prev
        
        self.prev_state = state
        return self.prev_state
    
    def agent_end(self,reward):
        target_update = reward
        Q_prev = self.Q_table[self.prev_state][self.prev_action] #Take the Q-value from the 
        Q_prev += self.alpha*(target_update - Q_prev)
        self.Q_table[self.prev_state][self.prev_action] = Q_prev


if __name__ == '__main__':

    # initiate environment
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs = env.reset()
    
    ql = QLearning(env,0.7,0.95,0.05)
    # action = env.observation_space
    reward_per_episode = []
    #Number of episodes to loop 
    for i in range(8000):
    #Initialize the agent 
        action = env.action_space.sample()
        obs,_,_,_ = env.step(action)
        if obs not in ql.Q_table:
            ql.initialize_q_values(obs)
            
        episode_reward = 0
        episode_score = 0 
        # iterate
        while True:
            # Select next action
            action = ql.agent_initialize(obs)
    #         action = env.action_space.sample()  # for an agent, action = agent.policy(observation)
            # Appy action and return new observation of the environment
            obs, _, done, info = env.step(action)
            reward = ql.give_reward(done)
            if obs not in ql.Q_table:
                ql.initialize_q_values(obs)
                
            obs = ql.agent_step(reward,obs)
            
            # Render the game
            os.system("clear")
            sys.stdout.write(env.render())
            print(f'Episode {i}')
            # if i < 990:
            time.sleep(0.0) # FPS
            # else:
                # time.sleep(0.2)
            # tot_rew += reward

            # If player is dead break
            episode_reward += reward
            if done:
                ql.agent_end(reward)
                break

            # if i%10 == 0:
            #     save_json(ql.Q_table)
        reward_per_episode.append(episode_reward)
        env.reset()


# reward_Qlearning = pd.DataFrame({'Rewards':reward_per_episode})
# print('Saving reward table to CSV!')
# reward_Qlearning.to_csv('/home/sauraj/Desktop/qlearning_rewards.csv')