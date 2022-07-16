******************PROJECT INFO******************
CartPole REINFORCE ALGO

Simple Demonstration of PG algorithm called REINFORCE(1992)

1) ENVIRONMENT
Open AI's Cartpole-V0 for training and Cartpole-V1 for testing.

2) ACTION SPACE
1 for left acceleration and 0 for right.

3) OBSERVATIONS SPACE
4 observations including velocity,pole angle with vertical,displacement and angular velocity.

4) POLICY
Neural Network Policy.

5) CREDIT ASSIGNMENT PROBLEM
Solved by using discounted rewards.

6) REWARDS
+1 for every step it takes.

7) POLICY GRADIENT-REINFORCE ALGORITHM
* Allow the neural network to play the game for some iterations and note down the gradients that make the action more likely.
* Calculate the discounted rewards after the above said iterations.
* Multiply the gradients with the discounted rewards.
* Perform the gradient descent step by using the mean of above calculated gradients.

The final result is really good. The network was able to perform with 100 percent reward gain for 200,500 and even 1500 episodes.

The model weight for 3 different configuration have been provided.

MODEL ARCHITECURE
model=Sequential([
	Input(shape=(4,))
	Dense(25,activation='relu')
	Dense(1,activation=NONE)
   )]
		      ***
