import numpy as np 

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
            return -10
    
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
            
        #Q-update equation
        target_update = reward + self.gamma*np.max(self.Q_table[state])
        Q_prev = self.Q_table[self.prev_state][self.prev_action] #Take the Q-value from the 
        Q_prev += self.alpha*(target_update - Q_prev)
        self.Q_table[self.prev_state][self.prev_action] = Q_prev
        
        self.prev_state = state
        # self.prev_action = action
        return self.prev_state
    
    def agent_end(self,reward):
        
        target_update = reward
        Q_prev = self.Q_table[self.prev_state][self.prev_action] #Take the Q-value from the 
        Q_prev += self.alpha*(target_update - Q_prev)
        self.Q_table[self.prev_state][self.prev_action] = Q_prev
