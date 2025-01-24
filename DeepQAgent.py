import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
import math

class DeepCartPoleBalancer(nn.Module):
	def __init__(self,n_actions,lr=0.03,drop_prob=0.2):
		super(DeepCartPoleBalancer,self).__init__()

		self.fc1 = nn.LazyLinear(128)
		self.fc2 = nn.Linear(128,256)
		self.fc3 = nn.Linear(256,n_actions)

		self.dropout = nn.Dropout(drop_prob)

		self.optimizer = torch.optim.Adam(self.parameters(), lr = lr)
		self.loss = nn.MSELoss()

	def forward(self,data):
		out = F.relu(self.fc1(data))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)

		return out

class BufferMemory():
	def __init__(self,mem_size,input_shape):
		self.mem_size = mem_size
		self.mem_pos = 0

		self.state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
		self.new_state_memory = np.zeros((self.mem_size,*input_shape),dtype=np.float32)
		self.action_memory = np.zeros(self.mem_size,dtype=np.int64)
		self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)
		self.terminal_memory = np.zeros(self.mem_size,dtype=bool)

	def store_transition(self,state,new_state,action,reward,terminated):
		index = self.mem_pos % self.mem_size

		self.state_memory[index] = state
		self.new_state_memory[index] = new_state
		self.action_memory[index] = action
		self.reward_memory[index] = reward
		self.terminal_memory[index] = terminated

		self.mem_pos += 1

	def sample_buffer(self,batch_size):
		max_mem = min(self.mem_pos,self.mem_size)
		batch = np.random.choice(max_mem,batch_size,replace=False)

		state = self.state_memory[batch]
		action = self.action_memory[batch]
		reward = self.reward_memory[batch]
		terminated = self.terminal_memory[batch]
		new_state = self.new_state_memory[batch]

		return state, action, reward, terminated, new_state


class DeepAgent():
	def __init__(self,env,mem_size = 5000,k = 0.005,epsilon_range = (0.01,1),alpha = 0.0003, gamma = 0.9,
					batch_size = 32,replace = 100):
		self.n_actions = env.action_space.n
		self.input_shape = env.observation_space.shape
		self.eps_decay_rate = k
		self.eps_min = epsilon_range[0]
		self.eps_max = epsilon_range[1]
		self.lr = alpha
		self.dc_factor = gamma

		self.learning_step = 0
		self.update_freq = replace
		self.batch_size = batch_size

		self.Online_Model = DeepCartPoleBalancer(n_actions = self.n_actions,lr = alpha)
		self.Target_Model = DeepCartPoleBalancer(n_actions = self.n_actions,lr=alpha)

		self.Memory = BufferMemory(mem_size,self.input_shape)

	def action(self,state):
		state = torch.tensor(state[np.newaxis,:],dtype=torch.float)
		prob_score = self.Online_Model(state)
		action = torch.argmax(prob_score).item()
		
		return action

	def update_target_model(self):
		if self.learning_step % self.update_freq == 0:
			self.Target_Model.load_state_dict(self.Online_Model.state_dict())

	def sample_memory(self):
		state, action, reward, terminated, new_state = self.Memory.sample_buffer(self.batch_size)

		state = torch.tensor(state)
		action = torch.tensor(action)
		reward = torch.tensor(reward)
		terminated = torch.tensor(terminated)
		new_state = torch.tensor(new_state)

		return state, action, reward, new_state, terminated

	def learn(self):
		if self.Memory.mem_pos < self.batch_size:
			return

		self.Online_Model.optimizer.zero_grad()

		self.update_target_model()

		states, actions, rewards, new_states, terminated = self.sample_memory()

		indices = torch.arange(self.batch_size)
		action_score = self.Online_Model(states)[indices, actions]
		
		with torch.no_grad():
			q_next = self.Target_Model(new_states).max(dim=1)[0]
			q_next[terminated] = 0.0
			target_score = rewards + self.dc_factor*q_next

		loss = self.Online_Model.loss(target_score,action_score)

		loss.backward()
		self.Online_Model.optimizer.step()

		self.learning_step += 1

		return loss.item() 

	def get_epsilon(self,ep):
		epsilon = self.eps_min + (self.eps_max - self.eps_min)* math.exp(-self.eps_decay_rate * ep)
		return epsilon

	def save_model(self,filename):
		torch.save(self.Online_Model.state_dict(),filename+"_O.pth")
		torch.save(self.Target_Model.state_dict(),filename+"_T.pth")
		print("V-------Models Saved-------V")

	def load_model(self,filename):
		self.Online_Model.load_state_dict(torch.load(filename+"_O.pth",weights_only=True))
		self.Target_Model.load_state_dict(torch.load(filename+"_T.pth",weights_only=True))
		print("V-------Model Loaded-------V")



def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)