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

# Function to create a keras sequential model
def model():
    model = Sequential([
        Input(shape=(4,)),
        Dense(25, activation='relu'),
        Dense(1, activation=None)
    ])
    return model

# Sigmoid helper activation function
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function to get the gradients from the Tape
def get_grads(obs):
    with tf.GradientTape() as tape:
        logits = model(obs.reshape(-1, 4), training=True) # Logits prediction from the model
        pred = Sigmoid(logits) # Getting the sigmoid probability
        action = random.choices([1, 0], weights=[pred, 1 - pred], k=1) # Weighted picking of actions
        y = 1.0 - np.float_(action) # Considering the current action as best and applying 0 if 1 and 1 if 0
        loss = binary_crossentropy(np.array([y]), logits, from_logits=True)
    grads = tape.gradient(loss, model.trainable_weights) # Get the gradients
    # optimizer.apply_gradients(zip(grads,model.trainable_weights))
    return grads, action

# Function to update the gradients by multiplying with the discounted rewards
def update_grads(gradients, reward):
    for i in range(len(gradients)):
        for j in range(len(gradients[i])):
            for k in range(len(gradients[i][j])):
                gradients[i][j][k] *= reward[i][j]
    return gradients

# Getting the mean of all collected gradients
def mean_grads(gradients):
    new_grad = [np.add(all_gradients[0][0], all_gradients[0][1])]
    for i in range(0, len(all_gradients)):
        for j in range(2, len(all_gradients[i])):
            new_grad = np.add(new_grad, all_gradients[i][j])
    return new_grad

# Function to convert the list of rewards to discounted rewards
def discount_rewards(reward, discount_rate):
    l = []
    for i in range(0, len(reward)):
        discounted_rewards = np.empty(len(reward[i]))
        cumulative_rewards = 0
        for step in reversed(range(len(reward[i]))):
            cumulative_rewards = reward[i][step] + cumulative_rewards * discount_rate
            discounted_rewards[step] = cumulative_rewards
        l.append(discounted_rewards)
    return l

# Optional function to normalize the rewards(DID NOT WORK WELL FOR ME)
def normalize_rewards(reward):
    mean_total=0
    count_mean=0
    for i in range(0,len(reward)):
        for j in range(0,len(reward[i])):
            mean_total+=reward[i][j]
            count_mean+=1
    mean_reward=mean_total/count_mean
    std_total=0
    count_rew=0
    for i in range(0, len(reward)):
        for j in range(0, len(reward[i])):
            std_total += (reward[i][j] - mean_reward) ** 2
            count_rew += 1
    std_reward = math.sqrt(std_total / (count_rew - 1))
    for i in range(0, len(reward)):
        for j in range(0, len(reward[i])):
            reward[i][j] = (reward[i][j] - mean_reward) / std_reward
    return reward

# The program execution starts from here
if __name__ == '__main__':
    episodes = 250 # Total no of episodes you want to train
    steps_per_episodes = 1200 #  Number of gradient descent steps you want
    gradient_collecting_episodes = 16 # The number of episodes you want to collect the gradients

    model = model() # Get the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
    optimizer = keras.optimizers.Adam(learning_rate=0.01)

    # Make and reset the environment
    env = gym.make("CartPole-v0")
    obs = env.reset()

    # Start the episodes loop
    for i in range(episodes):
        all_rewards = []
        all_gradients = []
        print("Episode", i, "Start")
        for j in range(gradient_collecting_episodes):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for k in range(steps_per_episodes):
                grads, action = get_grads(obs)
                obs, rewards, done, info = env.step(action[0])
                current_rewards.append(rewards)
                current_gradients.append(grads)
                if done:
                    print("Broken at:", k)
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        print("16 Done")
        all_rewards = discount_rewards(all_rewards, 0.99)
        #all_rewards=normalize_rewards(all_rewards)
        all_gradients = update_grads(all_gradients, all_rewards)
        all_gradients_total = mean_grads(all_gradients)
        final_gradients = all_gradients_total / (len(all_gradients) * 4) # Getting the mean of the total gradients
        optimizer.apply_gradients(zip(final_gradients[0], model.trainable_weights)) # Applying the new gradients to the optimizer
        print("Episode", i, "End")

    # Save the model weights after training is done
    model.save_weights(filepath="D:\PyCharm Projects\Reinforcement Learning\cartpole_model_50.h5")
