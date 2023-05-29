import gym_super_mario_bros
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3 import PPO
from random import *
if __name__=="__main__":
    model=PPO.load('testmodel1')
    env=gym_super_mario_bros.make('SuperMarioBros-v0')
    env=JoypadSpace(env,SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
    env = DummyVecEnv([lambda: env])
# 5. Stack the frames
    env = VecFrameStack(env, 4, channels_order='last')
    '''
    state=env.reset()
    for _ in range(1500):
        env.step([randint(1, 150)])
        env.render()
    '''
    state=env.reset()    
    while True:
        action,state=model.predict(state)
        #state,reward,done,info= env.step([env.action_space.sample()])        
        state,reward,done,info= env.step(action)        
        print(f"Pos: {info[0]['x_pos']},{info[0]['y_pos']}")
        print(f"Action: {SIMPLE_MOVEMENT[action[0]]}")
        env.render()
    env.close()

