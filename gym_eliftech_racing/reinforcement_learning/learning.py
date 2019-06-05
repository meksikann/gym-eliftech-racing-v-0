import numpy as np
# import pyglet
# from pyglet import gl
import argparse

# keras -------------->>>>>>>
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

# import training env
from gym_eliftech_racing.envs.racing_simple import RacingSimpleEnv

from rl.agents.dqn import DQNAgent
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

    model.summary()





    # from pyglet.window import key
    #
    # a = np.array([0.0, 0.0, 0.0])
    #
    #
    # def key_press(k, mod):
    #     global restart
    #     if k == 0xff0d: restart = True
    #     if k == key.LEFT:  a[0] = -1.0
    #     if k == key.RIGHT: a[0] = +1.0
    #     if k == key.UP:    a[1] = +1.0
    #     if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
    #
    #
    # def key_release(k, mod):
    #     if k == key.LEFT and a[0] == -1.0: a[0] = 0
    #     if k == key.RIGHT and a[0] == +1.0: a[0] = 0
    #     if k == key.UP:    a[1] = 0
    #     if k == key.DOWN:  a[2] = 0
    #
    #
    # env = RacingSimpleEnv()
    # env.render()
    # env.viewer.window.on_key_press = key_press
    # env.viewer.window.on_key_release = key_release
    # record_video = False
    # if record_video:
    #     from gym.wrappers.monitor import Monitor
    #
    #     env = Monitor(env, '/tmp/video-test', force=True)
    # isopen = True
    # while isopen:
    #     env.reset()
    #     total_reward = 0.0
    #     steps = 0
    #     restart = False
    #     while True:
    #         s, r, done, info = env.step(a)
    #         total_reward += r
    #         if steps % 200 == 0 or done:
    #             print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
    #             print("step {} total_reward {:+0.2f}".format(steps, total_reward))
    #             # import matplotlib.pyplot as plt
    #             # plt.imshow(s)
    #             # plt.savefig("test.jpeg")
    #         steps += 1
    #         isopen = env.render()
    #         if done or restart or isopen == False:
    #             break
    #
    # env.close()
