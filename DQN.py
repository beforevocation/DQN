# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
-------------------------------------------------
   File Name：     DQN.py  
   Description :  
   Author :       LJW
   date：          2018/1/28
-------------------------------------------------
   Change Activity:
                   2018/1/28 
-------------------------------------------------
"""

INITIAL_EPSILON = 0.5 # starting value of epsilon
ENV_NAME = 'CartPole-v0'

from collections import deque
import gym
import tensorflow as tf

def main():
    env = gym.make(ENV_NAME)
    agent = DQN(env)

class DQN():
    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        #初始化session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

    def create_Q_network(self):
        #这段代码确实需要补充一下神经网络的知识：MLP网络，中间层为20
        W1 = self.weight_variable([self.state_dim, 20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20, self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        #输入层
        self.input_layer = tf.place_holder("float", [None, self.state_dim])
        #隐藏层
        h_layer = tf.nn.relu(tf.matmul(self.input_layer, W1) + b1)
        #Q Value
        self.Q_value = tf.matmul((h_layer, W2) + b2)

    def weight_variable(self, shape):
        initial = tf.truncated_noraml(shape)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.variable(initial)

    def create_training_method(self):

