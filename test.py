import math
import random

import keras.optimizers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import gym

# Sigmoid Helper Function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Creating Cartpole gym environment
env = gym.make("CartPole-v1")
env._max_episode_steps = 1500
obs = env.reset()
done = False

# Creating a model skeleton and using the saved weights
model = Sequential([
    Input(shape=(4,)),
    Dense(25, activation='relu'),
    Dense(1, activation=None)
])
model.load_weights("D:\PyCharm Projects\Reinforcement Learning\cartpole_model_weights_25.h5")

# Episode Loop
total = 0
while (not done):
    env.render()
    logits = model(obs.reshape(-1, 4))
    pred = Sigmoid(logits)
    action = [1 if pred <= 0.5 else 0]
    print(action[0])
    obs, reward, done, info = env.step(action[0])
    total += reward
print(total, "DONE")
