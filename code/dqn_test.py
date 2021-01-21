from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from tensorflow import keras
import tensorflow as tf
import gym
import numpy as np


def test(model, save_path):
    # setting a seed
    seed = 42

    # setting the video save path
    video_save_path = '/'.join((save_path, 'video'))

    # initializing the environment
    env = make_atari("BreakoutNoFrameskip-v4")

    # Wrapping the env with deepmind wrapper
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env.seed(seed)

    # Adding the monitor as a wrapper to the environment
    env = gym.wrappers.Monitor(env, video_save_path, video_callable=lambda episode_id: True, force=True)

    # setting the return parameters
    n_episodes = 10
    rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        # Resetting the state for each episode
        state = np.array(env.reset())
        done = False

        while not done:
            # Choosing an action based on greedy policy
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_values = model.predict(state_tensor)
            action = np.argmax(action_values)

            # Perform action and get next state, reward and done
            state_next, reward, done, _ = env.step(action)
            state = np.array(state_next)

            # Update the reward observed at episode i
            rewards[i] += reward

    env.close()
    return rewards


def load_model(save_path):
    model_save_path = '/'.join((save_path, 'model'))
    return keras.models.load_model(model_save_path)


def main():
    breakout_save_path = '/DeepQNetwork-Atari-Breakout'

    # loading the model
    print('Loading the model...')
    model = load_model(breakout_save_path)

    # testing the model and returning rewards
    print('Testing the model...')
    rewards = test(model, breakout_save_path)

    print('Rewards: {}'.format(rewards))


if __name__ == "__main__":
    main()
