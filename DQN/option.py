class Option:

	def __init__(self):
		self.EPISODE_SIZE = 10000
		self.MAX_STEP_SIZE = 500

		self.TEST_EPISODE_SIZE = 10
		self.MAX_TEST_STEP_SIZE = 500

		self.MINIBATCH_SIZE = 32

		self.GAMMA = 0.9

		self.RMEMORY_LENGTH = 10000
		self.LEARNING_RATE = 1e-3

		self.STATE_NUM = 4
		self.ACTION_NUM = 2
        
		self.EPS_START = 0.9
		self.EPS_END = 0.01
		self.EPS_DECAY = 500000
        
		self.UPDATE_FREQ = 100
        
		self.TSCORE_THRESHOLD = 190
        
		self.HIDDEN1 = 30
		self.HIDDEN2 = 20

		self.learn_start = 1000

op = Option()