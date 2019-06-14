# CarRacing-v0
from gym_eliftech_racing.envs.racing_simple import RacingSimpleEnv
import numpy as np
import gym
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


# parse argiments ===================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='CarRacing-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()
#  =================================================================
ENV_NAME = args.env_name


# Get the environment and extract the number of actions ============
env = RacingSimpleEnv()
# env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model. ================================
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

#  critic model ===============================================================
action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


def main():
    weights_filename = 'weights/ddpg_{}_weights.h5f'.format(ENV_NAME)

    try:
        if args.mode == 'train':
            # Okay, now it's time to learn something! We visualize the training here for show, but this
            # slows down training quite a lot. You can always safely abort the training prematurely using
            # Ctrl + C.

            # load existing weights =============================================
            if args.weights:
                print('START WEIGHTS LOADED =============>>>>>>>>>>>>>')
                start_weights_filename = 'weights/' + args.weights
                agent.load_weights(start_weights_filename)
            #  ==================================================================

            checkpoint_weights_filename = 'checkpoint/ddpg_' + ENV_NAME + '_weights_{step}.h5f'
            log_filename = 'logs/ddpg_{}_log.json'.format(ENV_NAME)
            callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=2500)]
            callbacks += [FileLogger(log_filename, interval=100)]
            agent.fit(env, callbacks=callbacks, nb_steps=50000, log_interval=1000, visualize=True)

        elif args.mode == 'test':
            if args.weights:
                weights_filename = 'weights/' + args.weights

            agent.load_weights(weights_filename)
            agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

    finally:
        print('save model')
        agent.save_weights(weights_filename, overwrite=True)


if __name__ == '__main__':
    print('start file execution ...')
    main()
