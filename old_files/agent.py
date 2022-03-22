
import matplotlib.pyplot as plt 
import pandas as pd
from agents.QLearning import QLearning 
from agents.sarsa import SARSA
import os, sys
import gym
import time
import numpy as np
import text_flappy_bird_gym


def run_model(model):

    rl_model = model
    # action = env.observation_space
    reward_per_episode = []
    #Number of episodes to loop 
    for i in range(8000):
    #Initialize the agent 
        action = env.action_space.sample()
        obs,_,_,_ = env.step(action)
        if obs not in rl_model.Q_table:
            rl_model.initialize_q_values(obs)
            
        episode_reward = 0
        episode_score = 0 
        # iterate
        while True:
            # Select next action
            action = rl_model.agent_initialize(obs)
    #         action = env.action_space.sample()  # for an agent, action = agent.policy(observation)
            # Appy action and return new observation of the environment
            obs, _, done, info = env.step(action)
            reward = rl_model.give_reward(done)
            if obs not in rl_model.Q_table:
                rl_model.initialize_q_values(obs)
                
            obs = rl_model.agent_step(reward,obs)
            
            # Render the game
            os.system("clear")
            sys.stdout.write(env.render())
            print(f'Episode {i}')
            time.sleep(0.0) # FPS

            # If player is dead break
            episode_reward += reward
            if done:
                rl_model.agent_end(reward)
                break

            # if i%10 == 0:
            #     save_json(ql.Q_table)
        reward_per_episode.append(episode_reward)
        env.reset()

if __name__ == "__main__":
    env = gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    obs = env.reset()

    ql = QLearning(env,0.8,0.95,0.05)
    run_model(ql)



# reward_Qlearning = pd.DataFrame({'Rewards':reward_per_episode})
print('Saving reward table to CSV!')
reward_Qlearning.to_csv('/home/sauraj/Desktop/qlearning_rewards.csv')
