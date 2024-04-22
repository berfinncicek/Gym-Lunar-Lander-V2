#model egitilir ve kaydedilir güncellenerek

# import gym
# from stable_baselines3 import A2C


# env = gym.make("LunarLander-v2",render_mode="human")

# model_path= "model/A2C-1710843076/500000.zip"
# model = A2C.load(model_path, env=env)

# total_timesteps= 500000
# train_freq=64
# log_interval = 10

# model.learn(total_timesteps=total_timesteps)
# new_model_path = "model/A2C-continued"
# model.save(new_model_path)

# env.close()

import gym
from stable_baselines3 import A2C

env = gym.make("LunarLander-v2", render_mode="human")

model_path = "model/A2C-1710843076/500000.zip"
model = A2C.load(model_path, env=env)

total_timesteps = 500000
train_freq = 64
log_interval = 10
save_interval = 10000  
scipy_out= 5246


for i in range(total_timesteps // save_interval):
    model.learn(total_timesteps=save_interval, reset_num_timesteps=False)
    new_model_path = f"model/A2C-continDD_{500000 + (i+1) * save_interval}.zip"
    model.save(new_model_path)
    print(f"model {new_model_path} basarili")

env.close()
