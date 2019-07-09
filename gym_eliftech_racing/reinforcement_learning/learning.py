# CarRacing-v0
from __future__ import division
import argparse

from PIL import Image
import numpy as np
from gym_eliftech_racing.envs.racing_simple import RacingSimpleEnv
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (96, 96)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='CarRacing-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
# env = gym.make(args.env_name)
env = RacingSimpleEnv()
np.random.seed(123)
env.seed(123)

nb_actions = 4  # hardcoded for now as we use discrete action space instead of continues

#
# 1 - turn left
# 2 - turn right
# 3 - press gas
# 4 - press break
#

print('actions', env.action_space.sample())
print('actions nb', nb_actions)


input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_dim_ordering() == 'tf':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_dim_ordering() == 'th':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_dim_ordering.')
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

weights_filename = 'weights/discrete_{}_weights.h5f'.format(args.env_name)
checkpoint_weights_filename = 'checkpoint/discrete_' + args.env_name + '_weights_{step}.h5f'
log_filename = 'discrete_{}_log.json'.format(args.env_name)

try:
    if args.mode == 'train':
         # load existing weights =============================================
        if args.weights:
            print('START WEIGHTS LOADED =============>>>>>>>>>>>>>')
            start_weights_filename = 'weights/' + args.weights
            dqn.load_weights(start_weights_filename)
        #  ==================================================================

        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        # callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        # callbacks += [FileLogger(log_filename, interval=100)]
        dqn.fit(env,  nb_steps=1750000, visualize=True)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)
    elif args.mode == 'test':

        if args.weights:
            weights_filename = args.weights
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)
finally:
    print('save model')
    dqn.save_weights(weights_filename, overwrite=True)

