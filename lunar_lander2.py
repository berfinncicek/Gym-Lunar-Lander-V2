import gym
from stable_baselines3 import PPO
import keyboard



env = gym.make("LunarLander-v2", render_mode = "human")

observation, info = env.reset()

model = PPO("MlpPolicy", env, verbose =1)
model.learn(total_timesteps=10000)

episodes = 10

for ep in range(episodes):
  
    observation, info = env.reset()
   
    
    done = False
    while not done:
         env.render()
         action = env.action_space.sample()
         observation, reward, terminated, truncated, info = env.step(action)

         if keyboard.is_pressed('q'):
            done = True

    if done:
        break        
env.close()

""" print("sample action", env.action_space.sample()) #0-idle,1-go left,2-go right,3-start the engine
print("observation space shape",env.observation_space.shape) #number of observation spaces
print("sample observation", env.observation_space.sample()) #takes a random sample from the observation space  
"""

'''


agent: uzay araci
cevre:  ay yuzeyi
eylem: ajanin secimleri
gözlem: cevreden geribildirim
odul: puan
hedef: en yuksek puan


'''