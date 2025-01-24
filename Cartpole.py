import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import os

from DeepQAgent import DeepAgent, plot_learning_curve

os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = gym.make("CartPole-v1")

n_games = 50_000
step = 0
best_score = -math.inf
scores = []
avg_scores = []
step_list = []
epsilons=[]

gen = 5
load_model_file = f"Models/Model_{gen-1}"
save_model_file = f"Models/Model_{gen}"
plot_file = f"Plots/final_plot_{gen}"

#Hyperparameters
alpha = 0.003		#Learning Rate
gamma = 0.99		#discount factor
eps_range = (0.1,0.1)
decay_r = 0.0005
mem_size = 10000
batch_size = 64
replace = 200	
target_avg_score = 200

Agent = DeepAgent(env,mem_size=mem_size,batch_size=batch_size,replace=replace,k=decay_r,epsilon_range = eps_range,alpha=alpha,gamma=gamma)

Agent.load_model(load_model_file)

print("Training Started")

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

			Agent.Memory.store_transition(obs,new_obs,action,reward,term or trunc)

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

		print(f"Episode : {game} Steps : {step} Score : {score} Best_Score : {best_score} Avg Score per 100 steps : {avg_score} Epsilon : {epsilon:.4f}")

		if avg_score >= target_avg_score and epsilon < 0.2:
			break

	env.close()

	Agent.save_model(save_model_file)

	plot_learning_curve(step_list,avg_scores,epsilons,plot_file)

except KeyboardInterrupt:
	print("Stopping now ...")

	Agent.save_model(save_model_file)
	plot_learning_curve(step_list,avg_scores,epsilons,plot_file+"_interrupted")