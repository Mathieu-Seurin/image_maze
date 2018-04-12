from utils import *
from DQNAgent import *
import gym

# Careful, debug prints a lot of stuff
logging.basicConfig(level=logging.INFO)

env = gym.make('CartPole-v0')
print(env.action_space)
n_actions=2

env.reset()
resolution = [get_screen(env).shape[2], get_screen(env).shape[3]]
write_json_config_file('dqn_cartpole_', input_resolution=resolution,
            n_out=n_actions, conv_shapes=[16, 32, 32], dense_shapes=[])
agent = DQNagent('dqn_cartpole_')


agent.train(env, 500, lr=0.001, epsilon_init=0.05, epsilon_schedule='exp', batch_size=128)
print('\nTest initial model')
agent.load('dqn_cartpole_1')
agent.test(env, 20, verbose=False)
print('\nTest trained model')
agent.load('dqn_cartpole_491')
agent.test(env, 20, verbose=False)
