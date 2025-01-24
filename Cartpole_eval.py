import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

from DeepQAgent import DeepAgent, plot_learning_curve

os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym.make("CartPole-v1",render_mode='human')

n_games = 100
step = 0
best_score = -math.inf
scores = []
avg_scores = []
step_list = []
epsilons=[]
gen = 5		#the generation of model to eval
model_file = f"Models/Model_{gen}"
plot_file = f"Plots/eval_plot_{gen}"

#Hyperparameters
alpha = 0.0003		#Learning Rate
gamma = 0.9			#discount factor
eps_range = (0,0)
decay_r = 0.0005
mem_size = 1000
batch_size = 32
replace = 100	

Agent = DeepAgent(env,mem_size=mem_size,batch_size=batch_size,replace=replace,k=decay_r,epsilon_range = eps_range,alpha=alpha,gamma=gamma)
Agent.load_model(model_file)

print("Evaluation Started")

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

		print(f"Episode : {game} Score : {score} Best_Score : {best_score} Avg Score per 100 steps : {avg_score} Epsilon : {epsilon}")

	env.close()

	plot_learning_curve(step_list,avg_scores,epsilons,plot_file)

except KeyboardInterrupt:
	print("Stopping now ...")

	plot_learning_curve(step_list,avg_scores,epsilons,plot_file+"interrupted")