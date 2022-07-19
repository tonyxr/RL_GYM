#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:55:35 2022

Actor and Critic will be modeled using one neural network (2 fully connected layers)
At each iteration, model take state as input, output action probabilities (for all legal actions) and critic value respectively

Policy function chooses action by analyze features of current state, will favor or disfavor certain features

on-policy value function: expected return start at s and always act based on policy pi
Train the model that chooses actions based on policy pi that max expected return 

"Actor": expanded from policy gradient, the policy gradient method is the "actor" in actor-critic methods, used to select actions
"Critic": estimated value function, criticize the actions made by actor

Steps:
    1. run agent on the env to collect training data per episode
    2. compute expected return for each iteration
    3. compute loss () for combined actor-critic model 
    4. compute gradients and update network parameters
    repeat 1 to 4 until training succeed or max episode timeout

@author: Xiaoru Shi
"""

import gym
import numpy as np
import collections
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from typing import Any, List, Sequence, Tuple
import statistics
import tqdm

"""
from IPython import display as ipythondisplay
from PIL import Image
from pyvirtualdisplay import Display
"""

env = gym.make("CartPole-v1")

seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

LEARNING_RATE = 0.5 #alpha, how much new info will override old info, high = only immediate recent will be remembered, low = nothing is learned
# learning rate should decrease over time
DISCOUNT = 0.99 #gamma, cumulative reward = gamma * existing_reward + new_reward high = future reward, patient; low = focus on immediate reward, impatient
# gamma decrease over time in case of positive end reward, gamma stay constant in this case for agents to keep being alive
EPISODE = 3000 # run for 3000 episodes
STEP_PER_EPISODE_MAX = 500 # termination timeout for the env is at 500

# epsilon --> exploration rate, decreases over time
# if p < epsilon: random action, else, pull best action, generate random value between min and max, then compare with current epsilon
MAX_EXPLORE_RATE = 0.4
MIN_EXPLORE_RATE = 0.001

"""Set a training standard for 'solved'"""
REWARD_BAR = 480
EPISODE_CRITERION = 100

"""from env, dimension for NN is always 2^x """
NUM_FEATURES = 4
NUM_ACTION = 2
NUM_UNIT = 128

eps = np.finfo(np.float32).eps.item()
huberLossModel = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


"""since need fully connected NN, need tensorflow keras as superclass"""
class ActorCriticModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        """keras.layers.Dense create 1 dense fully-connected layer"""
        # units --> positive int, dimensionality of output space
        # activation --> activation function to use, relu = f(x) = max(0, x)
        """NN layer initilization"""
        self.common = layers.Dense(NUM_UNIT, activation = "relu")
        self.actor = layers.Dense(NUM_ACTION)
        self.critic = layers.Dense(1)
        
        
    # input the features of current state into self.common layer, process with relu --> max(0, x)
    # tf.Tensor represent a multidimensional array of elements, with elements of uniform known datatype, and a shape
    # When writing a TensorFlow program, the main object that is manipulated and passed around is the `tf.Tensor`.
    """
    Feed feature value into common layer
    The returned value from common layer is passed into actor and critic layer
    Returned value from actor layer is action probability, returned value from critic layer is policy gradient value
    
    This function feed value to Tensorflow NN and return the feedback value from NN 
    input: tf.Tensor, output: tf.Tensor, tf.Tensor
    """
    def call(self, inputFeature):
        reluX = self.common(inputFeature)
        print("relu-x", reluX)
        print("actor output", self.actor(reluX))
        print("critic output", self.critic(reluX))
        return self.actor(reluX), self.critic(reluX)
    
    """
    reformat returned values from env to match with NN layer input, immediate reward is 1, int, done as binary 0 or 1
    state is float, return a tuple of 3 np.ndarray
    env_step
    input: np.ndarray, output: tuple consists 3 np.ndarray
    """
    def stepTranslator(self, action):
        childState, reward, done, info = env.step(action)
        
        childState = childState.astype(np.float32)
        reward = np.array(reward, np.int32)
        done = np.array(done, np.int32)
        print("child state, ", childState)
        print("reward, ", reward)
        print("done, ", done)
        
        return (childState, reward, done)
        
    """
    wrap the function as a TensorFlow executable operation to be executed with TensorFlow
    tf_env_step
    """
    def stepTFWrapper(self, action):
        return tf.numpy_function(self.stepTranslator, [action], [tf.float32, tf.int32, tf.int32])
        
    """
    initialState is randomized by env, returned by env.reset()
    All action prob array (32 bit float), reward array (32 bit int), and done signal array(32 bit int) are non-fixed tensorflow array
    runs all steps within a single episode
    run_episode
    """
    def episodeProcessor(self, initialState):
        # initialize action probability array, reward array and done signal array
        # this function also does update values to storage arrays (like the functionality of Q-table)
        actionProbArray = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)
        criticValueArray = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)
        rewardArray = tf.TensorArray(dtype = tf.int32, size = 0, dynamic_size = True)
        
        currState = initialState
        stateShape = initialState.shape
        actorOutput = None
        criticOutput = None
        
        """replace the inner loop in Q-learning case"""
        for i in tf.range(STEP_PER_EPISODE_MAX):
            # update and convert state into a batched tensor (batch size = 1)
            # tf.expand_dims: works as "insert" into tf.Tensor, Returns a tensor with a length 1 axis inserted at index axis.
            # returns a tensor with same data as input, with an additional dimension inserted at index specified 
            currState = tf.expand_dims(currState, 0)
            
            # input the current state for NN layers, get action probabilities and critic value through NN layers
            actorOutput, criticOutput = self.call(currState)
            
            # sample next action based on returned probabilistic distribution
            # tf.random.categorical draws sample from a categorical distribution
            # return drawn samples of shape [batch_size, num_samples]
            actionSampled = tf.random.categorical(actorOutput, 1)[0, 0]
            
            # tf.nn.softmax computes soft-argmax, after softmax, original vector array values will be converted proportionally to be between 0 and 1,
            # sum of all outputs from softmax add up to 1, output of tf.nn.softmax hence can be consider as collection of probability of each action as outputed by NN actor layer
            # equivalent to: softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis, keepdims=True)
            # return type: tf.Tensor --> can be accept as numpy ndarray, size of the array is same as the size of the list passed in
            actionChosenProb = tf.nn.softmax(actorOutput)
            
            # store log of action probability and critic value
            # tf.squeeze: works as "pop" for tf.Tensor
            criticValueArray = criticValueArray.write(i, tf.squeeze(criticOutput))
            actionProbArray = actionProbArray.write(i, actionChosenProb[0, actionSampled])
            
            # pass action to env, receive reward, state and done signal
            currState, reward, done = self.stepTFWrapper(actionSampled)
            # align the childState with previous sate
            currState.set_shape(stateShape)
            
            # write reward to array
            rewardArray = rewardArray.write(i, reward)
            
            # tf.cast() in this case cast done signal from tf.Tensor to tf.bool
            if tf.cast(done, tf.bool):
                break
        
        # swtich the tensorarrays to tf.Tensor and return
        actionProbArray = actionProbArray.stack()
        criticValueArray = criticValueArray.stack()
        rewardArray = rewardArray.stack()
            
        return actionProbArray, criticValueArray, rewardArray
    
    """ 
    G_t = sum_all_timestamp(DISCOUNT^(time then - initial time) * reward_time_then)
    hence as time progress, for time then - initial time = x, DISCOUNT^x exponentially decay with convergence toward 0, as DISCOUNT range from (0, 1)
    expected return simply implies that rewards now are better than rewards later. In a mathematical sense, it is to ensure that the sum of the rewards converges
    """
    def computeReturn(self, rewardArray, standardize = True):
        # get number of rewards in the input array
        numReward = tf.shape(rewardArray)[0]
        # container for expected returns
        returnArray = tf.TensorArray(dtype = tf.float32, size = numReward)
        
        # retrieve reward from array backwards
        rewardArray = tf.cast(rewardArray[::-1], dtype = tf.float32)
        # discounted return value container, constant value holder 
        discountedReturn = tf.constant(0.0)
        # format for discounted value
        returnValueShape = discountedReturn.shape
        
        for i in tf.range(numReward):
            discountedReturn = rewardArray[i] + DISCOUNT * discountedReturn
            discountedReturn.set_shape(returnValueShape)
            
            # update value in storage
            returnArray = returnArray.write(i, discountedReturn)
        returnArray = returnArray.stack()[::-1]
        
        # reformat so return sequence can be analyzed with standard normal distribution (mean = 0, unit standard deviation = 1)
        if standardize:
            returnArray = ((returnArray - tf.math.reduce_mean(returnArray)) / (tf.math.reduce_std(returnArray) + eps))
            
        return returnArray
    
    # total loss = actor_loss + critic_loss for hybrid actor-critic model
    # actor loss is policy gradients with critic as a state dependent baseline, computed with single-sample (per-episode) estimates
    # L_actor = - sum (log (policy_output * [return(s_t, a_t) - V(s_t)]))
    # L_critic uses Huber loss
    # advantage function: indicates not how good action in absolute sense, but how much better the action compare to other actions available
    # A(s, a) = Q(s, a) - V(s) while acting on certain given policy
    def computeLoss(self, actionProb, criticValue, returnValue):

        actionAdvantage = returnValue - criticValue
        
        actorLoss = -(tf.math.reduce_sum(tf.math.log(actionProb) * actionAdvantage))
        criticLoss = huberLossModel(criticValue, returnValue)
        
        print("actor loss", actorLoss)
        print("critic loss", criticLoss)
        return actorLoss + criticLoss
        
"""
loop iterate for every episode

for fancy progress keeping, use tqdm.trange(EPISODE)
"""
episodesReward: collections.deque = collections.deque(maxlen=EPISODE_CRITERION)

#@tf.function
def iterator():
    cartpoleModel = ActorCriticModel()
    gradientValue = None
    
    
    with tqdm.trange(EPISODE) as t:
        for i in t:
            # receive initial state from env, 
            initTFState = tf.constant(env.reset(), dtype = tf.float32)
            # initialize model and NN through Tensorflow
    
            with tf.GradientTape() as tape:
                actionProbArray, criticValueArray, rewardArray = cartpoleModel.episodeProcessor(initTFState)
            
                # convert reward to expected return
                returnArray = cartpoleModel.computeReturn(rewardArray)
            
                # convert format of values from env to tf.tensor
                actionProbArray, criticValueArray, returnArray = [tf.expand_dims(x, 1) for x in [actionProbArray, criticValueArray, returnArray]]
                
                # get overall loss = actorLoss + criticLoss
                totalLoss = cartpoleModel.computeLoss(actionProbArray, criticValueArray, returnArray)
            
            print("total loss", totalLoss)
            
            gradientValue = tape.gradient(totalLoss, cartpoleModel.trainable_variables)
            print("gradient value", gradientValue)
            optimizer.apply_gradients(zip(gradientValue, cartpoleModel.trainable_variables))
            print("variables", cartpoleModel.trainable_variables)
            print("reward array", rewardArray)
            episodeReward = int(tf.math.reduce_sum(rewardArray))
            
            
            print("episode reward,", episodeReward)
            episodesReward.append(episodeReward)
            averageReward = statistics.mean(episodesReward)
            #print("sum reward", sum(episodesReward))
            
            t.set_description(f'Episode {i}', i)
            t.set_postfix(episodeReward=episodeReward, averageReward=averageReward)
        
            if i % 10 == 0:
                print("Episode {i}: average reward: {averageReward}", i, averageReward)
                
            if averageReward > REWARD_BAR and i >= EPISODE_CRITERION:  
                break
            
            env.render()
                
    print(f'\nSolved at episode {i}: average reward: {averageReward:.2f}!')

if __name__ == "__main__":
    iterator()