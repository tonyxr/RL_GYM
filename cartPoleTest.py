#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cart Pole Classic Control Tester
worklog: 
    7/5: look into documentation of Gym classic control, source code of Cartpole Env on Github
        https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    7/6: Read articles regarding Reinforcement learning designs and implementation, design basic framework and method hierarchy
        https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56 --> Q-table, reinforcement learning implementation
        https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ --> general knowledge about hyper-parameters and formula imeplementations, not cartpole env
        https://gsurma.medium.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288  --> concerning about Cartpole, but used keras and tensorflow
        https://itnext.io/reinforcement-learning-with-q-tables-5f11168862c8 --> Q-table
    7/7: code each method and overall structure
    
    End goal is multi-agent reinforcment learning, might be able to use tensorflow and keras for storage, may not need Deep Reinforcement Learning(DQN)
    
    Implement based on the pseudocode of the algorithm, tryout on various environments, test out the result on environment for the paper. 

@author: Xiaoru Shi
"""

import numpy as np
import gym
env = gym.make("CartPole-v1")
from gym import spaces
#import pygame
from collections import deque
import random
import matplotlib.pyplot as plt

"""    
    Q Learning Steps:
        1. Initialize the Q-table by all zeros.
        2. Start exploring actions: For each state, select any one among all possible actions for the current state (S).
        3. Travel to the next state (S') as a result of that action (a).
        4. For all possible actions from the state (S') select the one with the highest Q-value.
        5. Update Q-table values using the equation.
        6. Set the next state as the current state.
        7. If goal state is reached, then end and repeat the process.
        learning_rate = 0.4, epsilon = / (episode * 0.03)
        
for continuous action space, use (1 - lr) * Q(s,a) + lr * (reward + DISCOUNT * Q'(s', a)) also works
        
!!!Now the discretionized state space is small, to improve the result, add more intervals and increase the # of episode to run, change learning rate decay accordingly
"""

LEARNING_RATE = 0.5 #alpha, how much new info will override old info, high = only immediate recent will be remembered, low = nothing is learned
# learning rate should decrease over time
DISCOUNT = 0.95 #gamma, cumulative reward = gamma * existing_reward + new_reward high = future reward, patient; low = focus on immediate reward, impatient
# gamma decrease over time in case of positive end reward, gamma stay constant in this case for agents to keep being alive
STEPS = 1000 # run for 3000 episodes

# epsilon --> exploration rate, decreases over time
# if p < epsilon: random action, else, pull best action, generate random value between min and max, then compare with current epsilon
MAX_EXPLORE_RATE = 1
MIN_EXPLORE_RATE = 0.001

num_action = 2
num_observation = 4

"""
Flow of the model

1. iterator is called, supporter is initialized
2. First for loop iterates for every episode
3. second for loop iterate for each step
4. Action passed to iterator from supporter (by explore or by policy)
    a. getAction determine by epsilon if to sample random or choose by Q-value
    b. 
5. step taken and executed by environment, observation state (result state), immediate reward, terminal signal and info passed back
6. local supporter use new reward to calculate new Q-value, update Q-table accordingly by state and action pair

"""
        
class cartpoleSupporter():
    def __init__(self):
        # at beginning, choose random action to explore
        # exploration rate decay later to help converge the chosen policy
        self.epsilon = MAX_EXPLORE_RATE
        # number of factors in observation space, in this case = 4
        # store "discrete" Q-table 
        self.bucket, self.QTable = self.createQTable()
        
        self.learningRate = LEARNING_RATE
    
    """
    Observation Space --> np.array, continuous state, need to cut the range up into evenly spaced numbers to make them discrete
    Each state is list of 4 float number: 
        0: cart position: -4.8 to 4.8
        1: Cart Velocity: -Inf to Inf
        2: Pole Angle: -24 radian to 24 radian 
        3. Pole Angular Velocity: -Inf to Inf 
        
    Action Space: Discrete(2)
        0: push cart to left
        1: push cart to right
        
    """
    def createQTable(self):
        """need another implementation for Q-table"""
        
        # every observation space range will be broke into 30 smaller ranges evenly spaced, when                  
        # although cart velocity and pole angular velocity have infinite range, but range of reasonable velocity should be small
        # we define range of [-5, 5] for both velocity
        # spaceBucket will be used to translate each continous state into discrete state and update its value
        spaceBucket = [np.linspace(-2.4, 2.4, 6), np.linspace(-3, 3, 6), np.linspace(-0.2095, 0.2095, 6), np.linspace(-3, 3, 6)]
        print("space bucket", spaceBucket)
        # originally q-table is empty, key format as tuple --> (state, action), 
        # expand as ([cart_position, cart_velocity, pole_angle, pole_angular_velocuty], action)
        # value of each element is a tuple --> (reward: int, childState: list(float), done: bool)
                
        QTable = {}
        print("q table", QTable)
        
        return spaceBucket, QTable
    
    """
    Compute the action to take in the current state.  
    if value < self.epsilon, random action, else, best policy action 
    
    if reach terminal, return None
    """
    def getAcion(self, state):
        
        probChosen = random.uniform(0, MAX_EXPLORE_RATE)
        print("prob chosen,", probChosen)
        print("epsilon,", self.epsilon)
        # based on bucket, translate observed state to match state in Q-table
        discrete_state = self.translateState(state)
        
        if probChosen <= self.epsilon:
            # random action
            print("random from 1")
            return random.randint(0, num_action - 1)
        else:
            # policy action as by max Q-value
            print("rational 1")
            return self.getActionFromQValue(discrete_state)
        return None
    
    """
    Returns max_action Q(state,action) over available actions, used for argmax
    """
    
    def chooseValueFromQValue(self, discrete_state):
        rewardList = []
        for i in range(0, num_action):
            if (discrete_state, i) in self.QTable.keys():
                rewardList.append(self.QTable[(discrete_state, i)][0])
        
        if len(rewardList) < num_action:
            rewardList.append(0)
        print("reward list", rewardList)
        return max(rewardList)
        """
        if len(rewardList) >= 1:
            return max(rewardList)
        elif len(rewardList) == 1:
            return max([rewardList[0], 0])
        else:
            return 0.0
        """
    
    """
    Compute the best legal action to take in a state. There is no technically illegal action, but can reach to terminal with immediate negative reward
    at each state, the action space will be same, with all actions availble to cartpole agent
    
    Only 2 actions to take, either 0: left or 1: right
    """
    def getActionFromQValue(self, discrete_state):
        rewardList = []
        actionList = []
    
        for i in range(0, 2):
            if (discrete_state, i) in self.QTable.keys():
                rewardList.append(self.QTable[(discrete_state, i)][0])
                actionList.append(i)
        print("reward list", rewardList)
        print("action list", actionList)
        if len(rewardList) > 2 or len(rewardList) == 0:
            print("random 2")
            
            return random.randint(0, 1)
        if len(rewardList) == 1:
            if rewardList[0] < 0 and actionList[0] == 1:
                return 0
            elif rewardList[0] < 0 and actionList[0] == 0:
                return 1
            elif rewardList[0] >= 0 and actionList[0] == 1:
                return 1
            else:
                return 0
        elif rewardList[0] > rewardList[1]:
            return actionList[0]
        elif rewardList[0] == rewardList[1]:
            return random.randint(0, 1)
        else:
            return actionList[1]
    
    """
    Since the action space is continous, do Q'(s',a) is more reasonable than do Q'(s',a')
    """
    
    """
    def chooseValueFromQValue(self, childState, action):
        if (childState, action) in self.QTable.keys():
            return self.QTable[(childState, action)][0]
        else:
            return 0.0
    """
    
    """
    store new value in Q-table
    
    reward function should be directly related to pole angle, goal is to minimize the pole angle
    """
    def updateValue(self, state, action, reward, childState, done):
        
        currQ = 0
        newValue = 0
        print((state, action))
        # index 0 is cumulative reward
        if self.QTable != {} and (state, action) in self.QTable.keys():
            currQ = self.QTable[state, action][0]
            
            newValue = currQ * (1 - self.learningRate) + self.learningRate * (reward + DISCOUNT * self.chooseValueFromQValue(childState))
        else:
            newValue = self.learningRate * (reward + DISCOUNT * self.chooseValueFromQValue(childState))
        
        """
        currValue = (1 - LEARNING_RATE) * currQ
        valueAdded = LEARNING_RATE * reward
        futureValue = self.chooseValueFromQValue(discreteState)
        # Q(s, a) = Q(s,a) + T(s, a, s')[R(s, a, s') + discount*V(s')]
        # = to Q(state, action) <-- (1-alpha)*Q(state, action) + alpha(reward + gamma*argmax(Q(next_state, next_action)))
        # = (1-alpha) * Q(state, action) + alpha * reward + alpha * gamma * argmax(Q(next_state, next_action))
        """
        
        print("updated value", newValue)
            
        # check if state exist in q-table, if not, add new state, if done, no childState
        if self.QTable != {} and (state, action) in self.QTable.keys():
            temp = list(self.QTable[(state, action)])
            print("before update", temp)
            temp[0] = newValue
            self.QTable[(state, action)] = tuple(temp)
            print("after update", self.QTable[(state, action)])
            return newValue
        else:
            
            #print(self.QTable)
            self.QTable[(state, action)] = (newValue, childState, done)
            return newValue
        
    """
    since we divided states into small intervals,
    original state from env is continuous, we need to find the correct interval a given state is in
    
    Each state has 4 float values, compare one to one
    !!! checked
    """
    def translateState(self, state):
        stateHolder = []
        
        # four factor in each state, match one to one with state in q-table
        for i in range(4):
            minDifference = float("inf")
            valueChosen = 0
            
            for element in self.bucket[i]:
                if abs(state[i] - element) < minDifference:
                    minDifference = abs(state[i] - element)
                    valueChosen = element
                
            stateHolder.append(valueChosen)

        return tuple(stateHolder)
        
def iterator():
    """
    for _ in range(STEPS):
        # take random action
        observation, reward, done, info = env.step(env.action_space.sample())
        # call open the window
        env.render() 
        
    
        # after episode, reset to initial state
        if done:
            observation = env.reset()
    """
    localSupporter = cartpoleSupporter()
    
    countList = []
    episodeList = []
    stepRewardList = []
    rewardList = []
    Episode = 1
    count = 0
    for _ in range(STEPS):
        state = env.reset()
        print("Episode #", Episode)
        
        count = 0
        stepRewardList = []
        while True:
            
            print("step #", count)
            # find best action from supporter
            #action = random.randint(0, 1)
            action = localSupporter.getAcion(state)
            
            print("action taken", action)
            # immediate reward is +1 for every step lived
            childState, reward, done, info = env.step(action)
            print("reward original", reward)
            # compute new reinforced reward, use original state here, translate childState too
            reward = reward - abs(state[2]*10)
            # new state in list form
            # old state and new state in tuplen form 

            childStateDiscrete = localSupporter.translateState(childState)
            oldState = localSupporter.translateState(state)
            
            print("reward: ", reward)
            state = childState

            print("translated state, ", childStateDiscrete)
            print("old state,", oldState)
            print("state,", state)
            
            
            if childState[0] >= 4.8 or childState[0] <= -4.8 or childState[2] >= 0.418 or childState[2] <= -0.418:
                reward = -20
                # take action, see what childState is reached to, what's the reward, then update accordingly in q-table based on the original state
                newQ = localSupporter.updateValue(oldState, action, reward, childStateDiscrete, done)
                stepRewardList.append(newQ)
                break
            
            # if reach termination state
            if done and count <= 210:
                # take action, see what childState is reached to, what's the reward, then update accordingly in q-table based on the original state
                reward = -20
                newQ = localSupporter.updateValue(oldState, action, reward, childStateDiscrete, done)
                
                stepRewardList.append(newQ)
                break
            elif done:
                newQ = localSupporter.updateValue(oldState, action, reward, childStateDiscrete, done)
                stepRewardList.append(newQ)
                break
            else:
                newQ = localSupporter.updateValue(oldState, action, reward, childStateDiscrete, done)
                stepRewardList.append(newQ)
                # if continue, at end of each step iteration, decrease epsilon a bit, by end, should only select action by policy
        
            count += 1
            # show the anime
            env.render()
        
        if localSupporter.epsilon > MIN_EXPLORE_RATE:
            localSupporter.epsilon = localSupporter.epsilon/(Episode * 0.01)
        
        countList.append(count)
        episodeList.append(Episode)
        print("sum reward", sum(stepRewardList))
        rewardList.append(sum(stepRewardList)/len(stepRewardList))
        
        
        if localSupporter.learningRate < 5:
            localSupporter.learningRate *= 0.9995
        
        Episode += 1
    
    print("average reward, ", sum(rewardList)/len(rewardList))
        #print(localSupporter.QTable)
    plt.plot(episodeList, countList, label="duration for each trial")
    plt.plot(episodeList, rewardList, label="average reward")
    
   
    plt.show()
    print(dict(sorted(localSupporter.QTable.items())))

if __name__ == '__main__':
    iterator()
env.close()


