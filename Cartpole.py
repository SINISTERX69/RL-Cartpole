import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

from DeepQAgent import DeepAgent, plot_learning_curve

#Set this environment variable only if the console prompts you to 
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#initialize the environment
env = gym.make("CartPole-v1")

n_games = 50_000	#Number of games
best_score = -math.inf	#Track the current best score. 500 is the max
scores = []		#Store score of each episode
avg_scores = []		#Store the average score over last 100 episodes
step_list = []		#Store total learning steps after every episode
epsilons=[]		#Store the value of epsilon after every episode

gen = 5		#Specify the generation of model
load_model_file = f"Models/Model_{gen-1}"	#Load previous generation model
save_model_file = f"Models/Model_{gen}"		#specify the filename for new gen model
plot_file = f"Plots/final_plot_{gen}"		#name of the plot to store results

#Hyperparameters
alpha = 0.003		#Learning Rate
gamma = 0.99		#discount factor
eps_range = (0.1,0.1)	#Specify the range of epsilon
decay_r = 0.0005	#Epsilon decay rate (Exponential Decay)
mem_size = 10000	#The size of memory buffer. Greater the better
batch_size = 64		#Batch size for neural network
replace = 200		#Update frequency of target network
target_avg_score = 200	#Target average score for early stopping

#Initialize the agent
Agent = DeepAgent(env,mem_size=mem_size,batch_size=batch_size,replace=replace,k=decay_r,epsilon_range = eps_range,alpha=alpha,gamma=gamma)
#Load weights of previous gen model for both networks
Agent.load_model(load_model_file)

print("Training Started")
#Try except block to handle Manual Early Stopping
try:
	for game in range(1,n_games+1):
		GameOver = False
		obs , _ = env.reset()
		score = 0

		epsilon = Agent.get_epsilon(game)

		while not GameOver:
			#Epsilon Greedy Algorithm for exploration 
			if np.random.uniform(0,1) < epsilon:
				action = env.action_space.sample()
			else:
				action = Agent.action(obs)


			new_obs, reward, term, trunc, _ = env.step(action)
			score += reward

			#Load data to memory (numpy array)
			Agent.Memory.store_transition(obs,new_obs,action,reward,term or trunc)
			#Optimize the network
			Agent.learn()
			
			step+=1
			
			obs = new_obs
			GameOver = term or trunc

		scores.append(score)
		epsilons.append(epsilon)
		step_list.append(step)

		avg_score = np.mean(scores[-100:])
		avg_scores.append(avg_score)

		if best_score < score:
			best_score = score
		#Logging
		print(f"Episode : {game} Steps : {step} Score : {score} Best_Score : {best_score} Avg Score per 100 steps : {avg_score} Epsilon : {epsilon:.4f}")

		#Early Stopping
		if avg_score >= target_avg_score and epsilon < 0.2:
			break

	env.close()
	#Save Model
	Agent.save_model(save_model_file)
	#plot the results and save the plot
	plot_learning_curve(step_list,avg_scores,epsilons,plot_file)

except KeyboardInterrupt:
	print("Stopping now ...")
	#Save Model
	Agent.save_model(save_model_file)
	#plot results and save to disk
	plot_learning_curve(step_list,avg_scores,epsilons,plot_file+"_interrupted")
