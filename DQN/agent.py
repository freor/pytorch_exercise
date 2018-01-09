import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from environment import *
from replay_memory import *
from dqn import *

import option as op

class Agent(object):

	def __init__(self):
		self.epsilon = 0.9
		self.env = Environment()
		self.rmemory = Replaymemory()
		self.dqn = DQN()
		self.dqn.cuda()

		#self.criterion = nn.CrossEntropyLoss()
		self.optimizer - torch.optim.Adam(self.dqn.parameters(), lr=op.learning_rate, eps=1e-3)


	def select_action(self, state, is_train):

		# for TEST phase
		if not is_train:
			reutnr self.dqn.select_action(state)

		# epsilon-greedy
		if self.epsilon > random.random():
			return self.env.random_action()
		else:
			self.dqn.network_action(state)

	def train(self):

		state = self.env.state

		for e in range(EPOCH_SIZE):

			state = self.env.new_episode()

			for s in range(MAX_STEP_SIZE):

				action = self.select_action(state, True)
				next_state, reward, terminal, _ = self.act(action)

				self.rmemroy.add((state, next_state, reward, terminal))

				self.minibatch_train()

				if terminal is True:
					break

				state = next_state

	def play(self):

		state = None

		for t in range(TEST_EPISODE):

			state = self.env.new_episode()

			episode_reward = 0

			for s in range(MAX_TEST_STEP_SIZE):

				action = self.select_action(state, False)
				next_state, reward, terminal, _ = self.act(action)

				episode_reward += reward

				if terminal is True:
					print("Episode: %d, Total Reward: %d" % (e, episode_reward))
					break

				state = next_state

	def minibatch_train(self):
		state, next_state, reward, terminal = self.rmemory.minibatch(MINIBATCH_SIZE) # list

		max_qvalue = self.dqn.target_Q_value(next_state)
		qvalue = self.dqn.cnetwork(state)

		loss = np.sqrt(reward + GAMMA * max_qvalue - qvalue)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	# TODO
	'''
	def save(self):
		torch.save(self, './myModel.pth')

	def load(self):
		return torch.load('./myModel.pth')
	'''
