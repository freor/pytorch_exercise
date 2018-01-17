from collections import deque
import random
import numpy as np
import pdb
from option import *

# DEQUE version

class ReplayMemory(object):

	def __init__(self):
		self.memory = deque(maxlen=op.RMEMORY_LENGTH)

	def add(self, sample):
		sample = np.array(sample)
		
		self.memory.append(sample)

	def minibatch(self, size):
		assert (size <= len(self.memory))

		idx = random.sample(range(len(self.memory)), size)

		contents = []
		for i in idx:
			contents += [self.memory[i]]

		contents = np.array(contents)

		#state, action, next_state, reward, terminal
		state = contents[:, 0]
		action = contents[:, 1]
		next_state = contents[:, 2]
		reward = contents[:, 3]
		terminal = contents[:, 4]
		
		return state, action, next_state, reward, terminal

	@property
	def size(self):
		return len(self.memory)