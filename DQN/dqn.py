import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import numpy as np

import option as op


class QNetwork(object):

	def __init__(self):

		# inputs: state
		# outputs: action

		self.fc1 = nn.Linear(op.state_num, 60)
		self.fc2 = nn.Linear(60, 30)
		self.fc3 = nn.Linear(30, op.action_num)
		
		init.xavier_uniform(self.fc1.weight)
		init.xavier_uniform(self.fc2.weight)
		init.xavier_uniform(self.fc3.weight)

		init.constant(self.fc1.bias, 0)
		init.constant(self.fc2.bias, 0)
		init.constant(self.fc3.bias, 0)

		self.softmax = nn.Softmax()

	def foward(self, x):
		# inputs: state
		# outputs: action

		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		logits = x
		prob = self.softmax(logits)

		return logits, prob

class DQN(object):

	def __init__(self):
		self.cnetwork = QNetwork()
		self.tnetwork = QNetwork()

	def network_action(self, state):

		_, prob = self.cnetwork(state)

		return np.argmax(prob)

	def target_Q_value(self, state):

		logits, _ = self.tnetwork(state)

		return np.max(logits)




