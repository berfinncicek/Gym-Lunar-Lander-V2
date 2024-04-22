import gym
from stable_baselines3 import A2C
import os 
import time


models_dir = f"model/A2C-{int(time.time())}"
logdir =f"logs/A2C-{int(time.time())}"

if not os.path.exists(models_dir): 
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode = "human")

env.reset()

model = A2C("MlpPolicy", env, verbose =1,tensorboard_log=logdir)
TIMESTEPS = 10000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C") 
    model.save(f"{models_dir}/{TIMESTEPS*i}")

      
env.close()