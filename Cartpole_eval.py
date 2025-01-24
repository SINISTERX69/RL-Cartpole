import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

from DeepQAgent import DeepAgent, plot_learning_curve
#Add this env variable only if prompted by the console
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#initialize cartpole environment 
env = gym.make("CartPole-v1",render_mode='human')

n_games = 100			#Number of games to track metrics
step = 0			#Total steps counter
best_score = -math.inf		#Track best score
scores = []			#Store scores per episode
avg_scores = []			#Average scores per 100 episodes
step_list = []			#Store total steps after episode
epsilons=[]			#Store Epsilon Value
gen = 5				#The generation of model to eval
model_file = f"Models/Model_{gen}"	#Edit this to evaluate different agents. Leave out the _O.pth or _T.pth part as it is handled by class function.
plot_file = f"Plots/eval_plot_{gen}"

#Hyperparameters (Dummy Values)
alpha = 0.0003		#Learning Rate
gamma = 0.9		#Discount factor
eps_range = (0,0)	#Epsilon Range (Keep the range 0 to avoid unnecessary results)
decay_r = 0.0005	#Epsilon decay rate
mem_size = 0		#Memory size
batch_size = 32		#Batch Size
replace = 100		#Update Frequency of target network

#Initialize the model and load pre-trained weights
Agent = DeepAgent(env,mem_size=mem_size,batch_size=batch_size,replace=replace,k=decay_r,epsilon_range = eps_range,alpha=alpha,gamma=gamma)
Agent.load_model(model_file)

print("Evaluation Started")

#Try Except block to handle manual early stopping
try:
	for game in range(1,n_games+1):
		GameOver = False
		obs , _ = env.reset()
		score = 0

		epsilon = Agent.get_epsilon(game)

		while not GameOver:
			if np.random.uniform(0,1) < epsilon:
				action = env.action_space.sample()
			else:
				action = Agent.action(obs)

			new_obs, reward, term, trunc, _ = env.step(action)
			score += reward
			
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
		print(f"Episode : {game} Score : {score} Best_Score : {best_score} Avg Score per 100 steps : {avg_score} Epsilon : {epsilon}")

	env.close()
	
	#Plot and Save the Results
	plot_learning_curve(step_list,avg_scores,epsilons,plot_file)

except KeyboardInterrupt:
	print("Stopping now ...")
	#Plot and Save the Results
	plot_learning_curve(step_list,avg_scores,epsilons,plot_file+"interrupted")
