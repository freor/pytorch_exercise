from collections import deque
import randoms

import option as op


class ReplayMemory(object):

	'''
	def __init__(self):
		self.memory = deque(maxlen=1000)

	def add(self, sample):
		self.memory.append(sample)

	def minibatch(self, size):
		assert (size < len(self.memory))

		idx = random.sample(range(len(self.memory)), size)

		contents = []
		# TODO: range vs xrange ?
		for i in range(idx):
			contents += [self.memory[i]]

		return contents
	'''