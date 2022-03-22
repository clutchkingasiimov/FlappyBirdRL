
from agents.QLearning import QLearning 
from agents.sarsa import SARSA
import os, sys
import time

#SARSA trainer 
def run_sarsa(env,alpha,gamma,epsilon,episodes):

    sarsa = SARSA(env,alpha,gamma,epsilon)
    # action = env.observation_space

    #Store the tracking variables in a list
    score_per_episode = []
    reward_per_episode = []
    episode_length = []
    #Number of episodes to loop 
    for i in range(episodes):
    #Initialize the agent 
        action = env.action_space.sample()
        obs,_,_,_ = env.step(action) #State, (S)
        if obs not in sarsa.Q_table:
            sarsa.initialize_q_values(obs)
            
        #Track the reward/episode and the length/episode
        reward_episode = 0
        length_ep = 0

        #Initialize the agent and obtain the first action
        action = sarsa.agent_initialize(obs,eps_decay=0.001) #Initial action, (A)

        # iterate
        while True:
            # Select next action
    #         action = env.action_space.sample()  # for an agent, action = agent.policy(observation)
            # Appy action and return new observation of the environment
            obs, _, done, info = env.step(action) 
            reward = sarsa.give_reward(done) #Obtain reward, (R)
            if obs not in sarsa.Q_table:
                sarsa.initialize_q_values(obs)
                
            action = sarsa.agent_step(reward,obs,eps_decay=0.001) #Update to next state (S'), Choose next action (A')
            length_ep += 1 #Episode counter 

            # Render the game
            os.system("clear")
            sys.stdout.write(env.render())
            print(f'Episode {i}')
            print('SARSA')
            time.sleep(0.0) # FPS

            # If player is dead break
            reward_episode += reward
            if done:
                sarsa.agent_end(reward)
                break
        env.reset()
        score_per_episode.append(info['score']) #Save the scores 
        reward_per_episode.append(reward_episode) #Save the rewards 
        episode_length.append(length_ep) #Save the episodes 

    return episode_length,score_per_episode,reward_per_episode,sarsa.Q_table

#Q-Learning trainer 
def run_qlearning(env,alpha,gamma,epsilon,episodes):

    ql = QLearning(env,alpha,gamma,epsilon)
    # action = env.observation_space

    #Store the tracking variables in a list 
    score_per_episode = []
    reward_per_episode = []
    episode_length = []
    #Number of episodes to loop 
    for i in range(episodes):
    #Initialize the agent 
        action = env.action_space.sample()
        obs,_,_,_ = env.step(action)
        if obs not in ql.Q_table:
            ql.initialize_q_values(obs)
            
        #Track the reward/episode and length/episode
        reward_episode = 0
        length_ep = 0  
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
            length_ep += 1 #Episode counter 

            # Render the game
            os.system("clear")
            sys.stdout.write(env.render())
            print(f'Episode {i}')
            print('Q-Learning')
            time.sleep(0.0) # FPS

            # If player is dead break
            reward_episode += reward
            if done:
                ql.agent_end(reward)
                break

        env.reset()
        score_per_episode.append(info['score']) #Save the scores 
        reward_per_episode.append(reward_episode) #Save the rewards 
        episode_length.append(length_ep) #Save the episodes 

    return episode_length,score_per_episode,reward_per_episode,ql.Q_table

