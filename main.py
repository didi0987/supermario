import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack,GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
env=gym_super_mario_bros.make('SuperMarioBros-v0')
env=JoypadSpace(env,SIMPLE_MOVEMENT)
env=GrayScaleObservation(env,keep_dim=True)
env=DummyVecEnv([lambda:env])
env=VecFrameStack(env,4,channels_order='last')
model =PPO('CnnPolicy',env,tensorboard_log='./Logs',verbose=1,learning_rate=0.000001,n_steps=512)
model.learn(total_timesteps=100000)
done=True
for step in range(15000):
    if done:
        state=env.reset()
    state, reward, done, info=env.step(env.action_space.sample())
    print(f"{SIMPLE_MOVEMENT[env.action_space.sample()]}")
    print(f"{info['x_pos']},{info['y_pos']}")
    #plt.imshow(state)
    env.render()
env.close() 