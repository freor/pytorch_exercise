import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from environment import *
from replay_memory import *
from dqn import *
import pdb
from option import *

from copy import deepcopy
import math

class Agent(object):

	def __init__(self):
		self.global_step = 0
		self.env = Environment()
		self.rmemory = ReplayMemory()
		self.dqn = DQN()
		self.dqn.cuda()
		self.target = deepcopy(self.dqn)
		#self.target.cuda()
		self.eps_threshold = op.EPS_START

		#self.criterion = torch.nn.SmoothL1Loss()
		self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=op.LEARNING_RATE, eps=1e-3)

	def select_action(self, state, is_test):

		state = Variable(torch.cuda.FloatTensor([state]))

		# for TEST phase
		if is_test:
			return self.dqn.max_action(state)
        
		self.eps_threshold = op.EPS_END + (op.EPS_START - op.EPS_END) * math.exp(-1. * self.global_step / op.EPS_DECAY)
		self.global_step += 1

		# epsilon-greedy
		if self.eps_threshold > random.random():
			return self.env.random_action()
		else:
			return self.dqn.max_action(state)

	def train(self):

		state = self.env.state

		for e in range(op.EPISODE_SIZE):

			state = self.env.new_episode()
            
			episode_reward = 0
			for s in range(op.MAX_STEP_SIZE):

				action = self.select_action(state, False)
				next_state, reward, terminal, _ = self.env.act(action)
				episode_reward += reward

				self.rmemory.add([state, action, next_state, reward, terminal])
                
				if self.global_step % op.UPDATE_FREQ == 1:
					self.target = deepcopy(self.dqn)
                
				self.minibatch_train()

				if terminal is True:
					if e % 50 == 1:
						print("Episode: %d, Epsilon: %f, Total Reward: %d" % (e, self.eps_threshold, episode_reward))
					break

				state = next_state
            
			if e % 50 == 1:
				finish = self.play()
				if finish:
					return

	def play(self):

		state = None
        
		avg_reward = 0
		epi_count = 0

		for t in range(op.TEST_EPISODE_SIZE):

			state = self.env.new_episode()

			episode_reward = 0

			for s in range(op.MAX_TEST_STEP_SIZE):

				action = self.select_action(state, True)
				next_state, reward, terminal, _ = self.env.act(action)

				episode_reward += reward

				if terminal is True:
					avg_reward += episode_reward
					epi_count += 1
					print("TEST", episode_reward)
					break

				state = next_state
                
		avg_reward /= epi_count
		if avg_reward >= op.TSCORE_THRESHOLD:
			print("TRAINING FINISHED:", "TEST Avg Reward: %d" % (avg_reward))
			return True
		else:
			print("TEST Avg Reward %d" % (avg_reward))
			return False
            

	def minibatch_train(self):
		# if rmemroy size is less than minibatch size, then not train.
		if self.rmemory.size < op.MINIBATCH_SIZE or self.rmemory.size < op.learn_start:
			return

		state, action, next_state, reward, terminal = self.rmemory.minibatch(op.MINIBATCH_SIZE) # numpy
		yj = Variable(torch.cuda.FloatTensor(deepcopy(reward).tolist()))
		_next_state = Variable(torch.cuda.FloatTensor(next_state.tolist()))
		_t = terminal * 1.0
		_t = Variable(torch.cuda.FloatTensor(terminal.tolist()))
		_yj = self.target.max_qvalue(_next_state) * (- _t + 1.0) * op.GAMMA
		yj = _yj + yj
        
		state = Variable(torch.cuda.FloatTensor(state.tolist()))
		action = Variable(torch.cuda.LongTensor(action.tolist())).view(-1, 1)
		qvalue = self.dqn.action_qvalue(state, action).view(-1)

		loss = yj.sub(qvalue).pow(2)
		# PAST PROBLEM: (32) - (32x1) => (32x32) so, the loss is 600, in normal the loss is 1.01.
		loss = loss.mean()
		self.optimizer.zero_grad()
		loss.backward()

		self.optimizer.step()


