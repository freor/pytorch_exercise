import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

import numpy as np

from option import *

'''
class QNetwork(nn.Module):

	def __init__(self):
		super(QNetwork, self).__init__()

		# inputs: state
		# outputs: action

		self.fc1 = nn.Linear(op.STATE_NUM, 60)
		self.fc2 = nn.Linear(60, 30)
		self.fc3 = nn.Linear(30, op.ACTION_NUM)
		
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
'''

class DQN(nn.Module):

	def __init__(self):
		super(DQN, self).__init__()

		# inputs: state
		# outputs: action

		self.fc1 = nn.Linear(op.STATE_NUM, op.HIDDEN1)
		self.fc2 = nn.Linear(op.HIDDEN1, op.HIDDEN2)
		self.fc3 = nn.Linear(op.HIDDEN2, op.ACTION_NUM)
		
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

	def max_action(self, state):

		_, prob = self.foward(state)

		_, index = torch.max(prob, 1)

		return index.cpu().data[0]

	def max_qvalue(self, state):

		logits, _ = self.foward(state)

		m, _ = torch.max(logits, 1)

		return m

	def action_qvalue(self, state, action):
        
		logits, _ = self.foward(state)
        
		a_qvalue = logits.gather(1, action)
        
		return a_qvalue

        