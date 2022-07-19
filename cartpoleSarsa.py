#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:41:51 2022

SARSA method

@author: Xiaoru Shi
"""

import numpy as np
import gym
env = gym.make("CartPole-v1")
from gym import spaces
#import pygame
import random
import matplotlib.pyplot as plt

LEARNING_RATE = 0.5 #alpha, how much new info will override old info, high = only immediate recent will be remembered, low = nothing is learned
# learning rate should decrease over time
DISCOUNT = 0.95 #gamma, cumulative reward = gamma * existing_reward + new_reward high = future reward, patient; low = focus on immediate reward, impatient
# gamma decrease over time in case of positive end reward, gamma stay constant in this case for agents to keep being alive
STEPS = 3000 # run for 3000 episodes

# epsilon --> exploration rate, decreases over time
# if p < epsilon: random action, else, pull best action, generate random value between min and max, then compare with current epsilon
MAX_EXPLORE_RATE = 0.4
MIN_EXPLORE_RATE = 0.001

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
        
        probChosen = random.uniform(0, 1)
        print("prob chosen,", probChosen)
        print("epsilon,", self.epsilon)
        # based on bucket, translate observed state to match state in Q-table
        discrete_state = self.translateState(state)
        
        if probChosen <= self.epsilon:
            # random action
            print("random from 1")
            return random.randint(0, 1)
        else:
            # policy action as by max Q-value
            print("rational 1")
            return self.getActionFromQValue(discrete_state)
        return None
    
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
    def chooseValueFromQValue(self, childState, action):
        if (childState, action) in self.QTable.keys():
            return self.QTable[(childState, action)][0]
        else:
            return 0.0
    
    """
    store new value in Q-table
    
    reward function should be directly related to pole angle, goal is to minimize the pole angle
    """
    def updateValue(self, state, action, reward, childState, done):
        
        currQ = 0
        newValue = 0
        print((state, action))
        next_action = self.getAcion(childState)

        # index 0 is cumulative reward
        if self.QTable != {} and (state, action) in self.QTable.keys():
            currQ = self.QTable[state, action][0]
            newValue = currQ * (1 - self.learningRate) + self.learningRate * (reward + DISCOUNT * self.chooseValueFromQValue(childState, next_action))
        else:
            newValue = self.learningRate * (reward + DISCOUNT * self.chooseValueFromQValue(childState, next_action))
        
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
            if done:
                # take action, see what childState is reached to, what's the reward, then update accordingly in q-table based on the original state
                reward = -20
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
            localSupporter.epsilon = localSupporter.epsilon/(Episode * 0.04)
        
        countList.append(count)
        episodeList.append(Episode)
        print("sum reward", sum(stepRewardList))
        rewardList.append(sum(stepRewardList)/len(stepRewardList))
        
        
        if localSupporter.learningRate < 5:
            localSupporter.learningRate *= 0.9995
        
        Episode += 1
        
        #print(localSupporter.QTable)
    plt.plot(episodeList, countList, label="duration for each trial")
    plt.plot(episodeList, rewardList, label="average reward")
   
    plt.show()
    print(dict(sorted(localSupporter.QTable.items())))

if __name__ == '__main__':
    iterator()
env.close()



        
        