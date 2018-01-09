from environment import *

class Agent(object):

	def __init__(self):
		self.epsilon = 0.9
		self.env = Environment()

	def select_action(self):
		# epsilon-greedy
		if self.epsilon > random.random():
			return self.env.random_action()
		else:
			



	def train(self):

	def play(self):

	def save(self):

	def load(self):
