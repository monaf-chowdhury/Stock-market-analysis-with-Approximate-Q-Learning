# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 04:07:22 2022

@author: monaf
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from Linear_Model import LinearModel
from Stock_Market_Enviroment import StockEnv
from Agent import Agent

from datetime import datetime  
import os
import pickle

from sklearn.preprocessing import StandardScaler
np.set_printoptions(suppress=True)


def get_data():
    df = pd.read_csv('Final Stock Data.csv')
    df.drop(['Date'],inplace = True,axis = 1 )
    return df

def scaling(env):
    list_of_states= []
    done = False
    while not done:
        action = np.random.choice(env.action_space)
        state, reward, done, info =env.step(action)

        list_of_states.append(state)

    scaler=StandardScaler()
    scaler.fit(list_of_states)
    return scaler
        
def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def play_one_episode(agent, env, is_train,scaler):
    
    temp_test_action = []
    temp_reward_array=[]
    
    state = env.reset() # get the initial state 
    state = scaler.transform([state]) # scale the state vector
    done = False

    while not done:
        action = agent.act(state) # get the next action
        
        if is_train == False:
            temp_test_action.append(action)
            
        next_state, reward, done, info = env.step(action) # perform the action
        temp_reward_array.append(reward)
        next_state = scaler.transform([next_state]) # scale the next state
        
        if is_train == True: # if the mode is training
            agent.train(state, action, reward, next_state, done) # Q-Learning with states' aggregation
        state = next_state # got to next state
        
     
    return info['current value'],temp_reward_array,temp_test_action

def Rewards(episode_rewards,num_episodes,initial_investment,path):
    episode_number=[]
    average_rewards_all_episodes = []
    
    for i in range(len(episode_rewards)):
        average_rewards_all_episodes.append(sum(episode_rewards[i])/len(episode_rewards[0]))
        episode_number.append(i+1)
    
    fin_reward = np.load(path)
    mean = np.mean(fin_reward)
    
    print("Maximum:            ", max(fin_reward))
    print("Minimum:            ", min(fin_reward))
    print("Mean:               ", np.mean(fin_reward))
    print("Standard Deviation: " ,np.std(fin_reward))
    
    
    roi = ((mean-initial_investment)/initial_investment)*100
    print("Return of Investment: {0:.4f}".format(roi))
    avg_reward =  np.sum(average_rewards_all_episodes)/num_episodes
    print(f"Average Return on All Episodes:{avg_reward}")
    #Plot rewards 
    
    plt.figure(figsize=(10,6))
    plt.bar(episode_number, list(average_rewards_all_episodes))
    plt.title("Average Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    return average_rewards_all_episodes
    
def Train_Test(data,is_train, num_episodes, initial_investment):
    episode_rewards=[]
    test_action =[]
    portfolio_value = [] # to collect the values at the end of episodes/epochs
    
    models_folder = 'linear_rl_model' # to store the Q-model prarameters
    rewards_folder = 'linear_rl_rewards' # to store the values of episodes
    
    make_directory(models_folder)
    make_directory(rewards_folder)
    
    env = StockEnv(data, initial_investment,3) # initialize the enviroment
    state_size = env.state_dimension # initialize state dimension
    action_size = len(env.action_space)  # initialize actions dimension
    agent = Agent(state_size, action_size) # initialize the agent's class
    scaler = scaling(env) # get the scaling parameters
    
    
    if is_train == False:
        #load previous scaler
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # remake the env with the test data
        env = StockEnv(data, initial_investment,3)

        agent.epsilon = 0.01
        agent.load(f'{models_folder}/linear.npz')

    for e in range(num_episodes):
        t0 = datetime.now()
        result = play_one_episode(agent, env, is_train, scaler)
        val,temp_reward_array = result[:2]
        if is_train == False:
            temp_test_action = result[-1]
            test_action.append(temp_test_action)
        
        dt = datetime.now() - t0
        
        episode_rewards.append(temp_reward_array) # every episodic reward is returned and stored in episode rewards
        print(f"episode: {e +1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val) 

    
    if is_train == True:

        agent.save(f'{models_folder}/linear.npz')
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler,f)

        plt.plot(agent.model.losses)
        plt.show()
    
    # save portfolio value for each episode
    if is_train == True:
        np.save(f'{rewards_folder}/train.npy', portfolio_value)
        np.save(f'{rewards_folder}/losses.npy', agent.model.losses)
        
    else:
        np.save(f'{rewards_folder}/test.npy', portfolio_value)
        return episode_rewards,test_action,env
    
    return episode_rewards

# Generating_Graphing_Data the test data 

def Generate_Graphing_Data (average_rewards_all_episodes,test_action,env):
    best_test_score_index = np.argmax(average_rewards_all_episodes)
    test_action_list = []
    
    for i in range(len(test_action[0])):
        test_action_list.append(env.action_list[test_action[best_test_score_index][i]])
    
    BX = [i[0] for i in test_action_list]
    GP = [i[1] for i in test_action_list]
    SQ = [i[2] for i in test_action_list]
    
    data = pd.read_csv('Final Stock Data.csv')
    n_timesteps,n_features = data.shape
    n_train = n_timesteps*3//4
    
    test_data=data.iloc[n_train+1:,[0,1,7,13]]
    test_data.reset_index(drop =True, inplace=True)
    
    test_data["BX"] = BX
    test_data["GP"] = GP
    test_data["SQ"] = SQ
    
    test_data.to_csv("Testing_Stock_Data.csv")
    test_data = pd.read_csv("Testing_Stock_Data.csv")
    test_data.drop(["Unnamed: 0"], axis=1, inplace= True)
    
    bx = list(test_data["BX"])
    bx = buy_sell(bx)
    test_data["BX"] = bx

    sq = list(test_data["SQ"])
    sq = buy_sell(sq)
    test_data["SQ"] = sq

    GP = list(test_data["GP"])
    GP = buy_sell(GP)
    test_data["GP"] = GP
    
    test_data["GP_sell"]=0
    test_data["GP_buy"]=0
    
    test_data["SQ_sell"]=0
    test_data["SQ_buy"]=0
    
    test_data["BX_sell"]=0
    test_data["BX_buy"]=0
    
    for i in range(len(test_data)):
        #GP
        if test_data["GP"][i]==0:
            test_data["GP_sell"][i] = test_data["GP_Price"][i]
        if test_data["GP"][i]==2:
            test_data["GP_buy"][i] = test_data["GP_Price"][i]
    
          
        #BX
        if test_data["BX"][i]==0:
            test_data["BX_sell"][i] = test_data["BX_Price"][i]    
        if test_data["BX"][i]==2:
            test_data["BX_buy"][i] = test_data["BX_Price"][i]
           
        #SQ
        if test_data["SQ"][i]==0:
            test_data["SQ_sell"][i] = test_data["SQ_Price"][i]
        if test_data["SQ"][i]==2:
            test_data["SQ_buy"][i] = test_data["SQ_Price"][i]
    
    return test_data

def buy_sell(List):
    length_of_list = len(List)
    first_index_of_zero, first_index_of_two = List.index(0), List.index(2)
    # 0: sell # 1: hold # 2: buy
    
    if first_index_of_zero < first_index_of_two:
        current_value, current_index = 0, first_index_of_zero
    else:
        current_value, current_index = 2, first_index_of_two

    if current_index>0:
        List[0:current_index]=[3]*(current_index)

    while True :
        try:
            next_value = 2 - current_value
            next_index = List.index(next_value, current_index)
            List[current_index+1:next_index] = [3]*(next_index-current_index-1)
            current_value, current_index = next_value, next_index
        except ValueError:
            List[current_index+1:length_of_list]=[3]*(length_of_list-current_index-1)
            break
    return List
            
def Graphing(test_data,name):
    price = name+'_Price'
    
    plt.figure(figsize=(16,6))
    plt.plot(test_data["Date"], test_data[price],"black")
    
    x = np.array(test_data["Date"])
    
    y = np.array(test_data[name+"_sell"]) # BX_sell
    plt.scatter(x=x[y>1], y=y[y>1])
    
    
    y=np.array(test_data[name+"_buy"])
    plt.scatter(x=x[y>0], y=y[y>0], c="red")
    
    plt.legend(["Price", "Sell" , "Buy"])
    plt.title(name)
    plt.xlabel("Date",fontsize=12)
    plt.ylabel("Price")
    plt.show()

##### After passing scaler ####### 
if __name__=="__main__":
    
    num_episodes_train = 1500 # epochs     
    num_episodes_test = 100       
    #batch_size = 32
    initial_investment = 10000

    data = get_data()
    n_timesteps,n_features = data.shape
    n_train = n_timesteps*3//4
    
    train_data = data.iloc[:n_train,:]    # 75% of the data for training
    test_data = data.iloc[n_train:,:]     # 25% of the data for testing    
    test_data.reset_index(inplace=True,drop=True) # setting the index from 0
    
    # Training on the data set with 1500 episodes for test_run
    episode_rewards = Train_Test(train_data,True, num_episodes_train, initial_investment)
    
    # Showing reward graphs on that 1500 episodes
    
    print("REWARDS:\n")
    average_rewards_all_episodes = Rewards(episode_rewards,num_episodes_train,initial_investment,(r"D:\Coding\Spyder\Project Stock Trading with RL\linear_rl_rewards\train.npy"))
    
    
    # Now the testing part # 
    episode_rewards,test_action,env = Train_Test(test_data,False, num_episodes_test, initial_investment)
    print("REWARDS:\n")
    average_rewards_all_episodes = Rewards(episode_rewards,num_episodes_test,initial_investment,(r"D:\Coding\Spyder\Project Stock Trading with RL\linear_rl_rewards\test.npy"))
    graphing_data = Generate_Graphing_Data (average_rewards_all_episodes,test_action,env)
    
    
    Graphing(graphing_data,'GP')
    Graphing(graphing_data,'BX')
    Graphing(graphing_data,'SQ')
    