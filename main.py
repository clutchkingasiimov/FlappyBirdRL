import pandas as pd
import gym
import numpy as np
from agents.trainer import run_sarsa, run_qlearning
import text_flappy_bird_gym

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


