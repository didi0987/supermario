import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv,VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
LOG_DIR='./Logs'
CHECKPOINT_DIR = './train'
class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
env=gym_super_mario_bros.make('SuperMarioBros-v3')
env=JoypadSpace(env,SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR,verbose=1,learning_rate=0.000001, n_steps=512) 
# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000,callback=callback)
#model.save('testmodel2')
'''
done=True
for step in range(15000):
    if done:
        state=env.reset()
    state, reward, done, info=env.step(env.action_space.sample())
    env.render()
env.close()'''

