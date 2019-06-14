import numpy as np
# import pyglet
# from pyglet import gl
import argparse
from PIL import Image

# keras -------------->>>>>>>
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

# import training env
from gym_eliftech_racing.envs.racing_simple import RacingSimpleEnv

# from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
ACTIONS_number = 3
WINDOW_LENGTH = 4

class GamePreprocessor(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize image and convert it to black/white
        processed_observation = np.array(img)
        return processed_observation.astype('uint8')  # save in memory with type

    def process_state(self, batch):
        processed_batch = batch.astype('float32') / 255
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)  # values smaler then -1 become -1 and values bigger then 1 become 1


# run manual racing
if __name__ == "__main__":

    # get command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--weights', type=str, default=None)
    arguments = parser.parse_args()

    env = RacingSimpleEnv()

    np.random.seed(234)
    env.seed(234)


    # build model
    input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
    model = Sequential()

    # check image ordering if 'tensorflow' or 'theano' and add first layer
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')

    # OPTION: 1  - add convolution layers
    # model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    # model.add(Activation('relu'))
    # model.add(Convolution2D(64, (3, 3), strides=(2, 2)))
    # model.add(Activation('relu'))
    # model.add(Flatten())
    # model.add(Dense(512))
    # model.add(Activation('relu'))
    # model.add(Dense(ACTIONS_number))
    # model.add(Activation('softmax'))

    # Option 2: deep network
    model = Sequential()
    model.add(Flatten(input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(WINDOW_LENGTH))
    # model.add(Activation('softmax'))

    model.summary()

    # load saved checkpoint weights
    if arguments.weights:
        print('Loading checkpoint weights to proceed with training.....')
        model.load_weights(arguments.weights)


    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    state_processor = GamePreprocessor()

    # policy = LinearAnnealedPolicy(
    #     EpsGreedyQPolicy(),
    #     attr='eps',
    #     value_max=1.,  # eps max
    #     value_min=0.1,  # eps min
    #     value_test=0.05,  # eps value during testing
    #     nb_steps=1000000  # eps steps
    # )
    #
    # # init RL agent, with model, gamma and cetra
    # dqn = DQNAgent(
    #     model=model,
    #     nb_actions=ACTIONS_number,
    #
    #     memory=memory,
    #     processor=state_processor,
    #     nb_steps_warmup=50000,
    #     gamma=.99,
    #     target_model_update=10000,
    #     train_interval=4,
    #     delta_clip=1.
    # )
    #
    # # compile DQN agent
    # dqn.compile(
    #     Adam(lr=.00025),
    #     metrics=['mae']
    # )

    cem = CEMAgent(model=model, nb_actions=ACTIONS_number, memory=memory, processor=state_processor,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
    cem.compile()

    # run agents mode
    if arguments.mode == 'train':
        env_name = 'car_racing'
        weights_file = 'dqn_{}_weights.h5f'.format(env_name)
        checkpoint_filename = 'dqn_' + env_name + '_weights_{step}.h5f'
        logs = 'dqn_{}_log.json'.format(env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_filename, interval=1000)]
        callbacks += [FileLogger(logs, interval=100)]

        # fit the aqn
        # dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)
        cem.fit(env, callbacks=callbacks, nb_steps=100000, visualize=True, verbose=2)

        # save weights
        # dqn.save_weights(weights_file, overwrite=True)
        cem.save_weights(weights_file, overwrite=True)

        #  evaluate algorithm
        # dqn.test(env, nb_episodes=10, visualize=True)


