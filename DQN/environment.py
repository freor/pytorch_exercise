import gym

class Environment(object):

	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.state = self.env.reset()

	def random_action(self):
		return env.action_space.sample()

	def new_episode(self):
		self.state = self.env.reset()

		return self.state

	def act(self, action):
		# next_state, reward, terminal, info
		return env.step(action)
