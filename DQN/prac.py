import gym
import cv2

# Make an environment instance of CartPole-v0.
env = gym.make('CartPole-v0')

# Before interacting with the environment and starting a new episode, you must reset the environment's state.
state = env.reset()

# Uncomment to show the screenshot of the environment (rendering game screens)
# env.render() 

# You can check action space and state (observation) space.
num_actions = env.action_space.n
state_shape = env.observation_space.shape
print(num_actions)
print(state_shape)

# "step" function performs agent's actions given current state of the environment and returns several values.
# Input: action (numerical data)
#        - env.action_space.sample(): select a random action among possible actions.
# Output: next_state (numerical data, next state of the environment after performing given action)
#         reward (numerical data, reward of given action given current state)
#         terminal (boolean data, True means the agent is done in the environment)
next_state, reward, terminal, info = env.step(env.action_space.sample())

print(next_state)
print(env.action_space.sample())