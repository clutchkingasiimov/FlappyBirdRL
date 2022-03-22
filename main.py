
import matplotlib.pyplot as plt 
import pandas as pd
from agents.QLearning import QLearning 
from agents.sarsa import SARSA
import os, sys
import gym
import time
import numpy as np
from agents.trainer import run_sarsa, run_qlearning
import text_flappy_bird_gym

# #SARSA
# def run_sarsa(env,alpha,gamma,epsilon,episodes):


#     sarsa = SARSA(env,alpha,gamma,epsilon)
#     # action = env.observation_space
#     score_per_episode = []
#     reward_per_episode = []
#     episode_length = []
#     #Number of episodes to loop 
#     for i in range(episodes):
#     #Initialize the agent 
#         action = env.action_space.sample()
#         obs,_,_,_ = env.step(action) #State, (S)
#         if obs not in sarsa.Q_table:
#             sarsa.initialize_q_values(obs)
            
#         reward_episode = 0
#         length_ep = 0

#         action = sarsa.agent_initialize(obs,eps_decay=0.001) #Initial action, (A)

#         # iterate
#         while True:
#             # Select next action
#     #         action = env.action_space.sample()  # for an agent, action = agent.policy(observation)
#             # Appy action and return new observation of the environment
#             obs, _, done, info = env.step(action) 
#             reward = sarsa.give_reward(done) #Obtain reward, (R)
#             if obs not in sarsa.Q_table:
#                 sarsa.initialize_q_values(obs)
                
#             action = sarsa.agent_step(reward,obs,eps_decay=0.001) #Update to next state (S'), Choose next action (A')
#             length_ep += 1 #Episode counter 

            
#             # Render the game
#             os.system("clear")
#             sys.stdout.write(env.render())
#             print(f'Episode {i}')
#             print('SARSA')
#             time.sleep(0.0) # FPS

#             # If player is dead break
#             reward_episode += reward
#             if done:
#                 sarsa.agent_end(reward)
#                 break

#             # if i%10 == 0:
#             #     save_json(ql.Q_table)
#         env.reset()
#         score_per_episode.append(info['score']) #Save the scores 
#         reward_per_episode.append(reward_episode) #Save the rewards 
#         episode_length.append(length_ep)

#     return episode_length,score_per_episode,reward_per_episode,sarsa.Q_table

# #Q-Learning
# def run_qlearning(env,alpha,gamma,epsilon,episodes):

#     ql = QLearning(env,alpha,gamma,epsilon)
#     # action = env.observation_space
#     score_per_episode = []
#     reward_per_episode = []
#     episode_length = []
#     #Number of episodes to loop 
#     for i in range(episodes):
#     #Initialize the agent 
#         action = env.action_space.sample()
#         obs,_,_,_ = env.step(action)
#         if obs not in ql.Q_table:
#             ql.initialize_q_values(obs)
            
#         reward_episode = 0
#         length_ep = 0  
#         # iterate
#         while True:
#             # Select next action
#             action = ql.agent_initialize(obs)
#     #         action = env.action_space.sample()  # for an agent, action = agent.policy(observation)
#             # Appy action and return new observation of the environment
#             obs, _, done, info = env.step(action)
#             reward = ql.give_reward(done)
#             if obs not in ql.Q_table:
#                 ql.initialize_q_values(obs)
                
#             obs = ql.agent_step(reward,obs)
#             length_ep += 1 #Episode counter 

            
#             # Render the game
#             os.system("clear")
#             sys.stdout.write(env.render())
#             print(f'Episode {i}')
#             print('Q-Learing')
#             time.sleep(0.0) # FPS

#             # If player is dead break
#             reward_episode += reward
#             if done:
#                 ql.agent_end(reward)
#                 break

#             # if i%10 == 0:
#             #     save_json(ql.Q_table)
#         env.reset()
#         score_per_episode.append(info['score']) #Save the scores 
#         reward_per_episode.append(reward_episode) #Save the rewards 
#         episode_length.append(length_ep)

#     return episode_length,score_per_episode,reward_per_episode,ql.Q_table


#Main 
if __name__ == "__main__":
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs = env.reset()

    gammas = np.linspace(0.1,1,10)  #Step-size sweep
    mean_rewards = []
    for gamma in gammas:
        # print(f'Step size: {step_size}')
        _,_,reward_per_stepsize,_ = run_sarsa(env,0.8,gamma,0.05,300)
        mean_rewards.append(np.sum(reward_per_stepsize)) #Append the rewards for each step-size 
    
    print('Model training done!')

    step_size_sweep = pd.DataFrame({'Gamma':gammas,
    'Mean Reward':mean_rewards})

    print('Saving Parameter Sweep for Gamma!')
    step_size_sweep.to_csv('/home/sauraj/Desktop/ps_g_sarsa.csv')

    # info_df = pd.DataFrame({'Rewards':reward_per_episode,'Score':score_per_episode,
    # 'Episode_Length':episode_length})

    # qtable_df = pd.DataFrame({'State':q_table.keys(),
    # 'Action':q_table.values()})

    # qtable_df = pd.DataFrame(q_table).T

    # print('Saving information dataframe to CSV!')
    # print('Saving Q table to CSV!')
    # info_df.to_csv('/home/sauraj/Desktop/info_df_qlearning.csv')
    # qtable_df.to_csv('/home/sauraj/Desktop/qtable_qlearning.csv')


