import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

env=gym_super_mario_bros.make('SuperMarioBros-v0')
env=JoypadSpace(env,SIMPLE_MOVEMENT)

done=True
for step in range(15000):
    if done:
        state=env.reset()
    state, reward, done, info=env.step(env.action_space.sample())
    env.render()
env.close()

