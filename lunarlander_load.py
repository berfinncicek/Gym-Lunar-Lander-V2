import gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="human")

models_dir = "model/PPO-1710841779"
model_path = f"{models_dir}/300000.zip"

model = PPO.load(model_path, env=env)

episodes = 10

episode_lengths = []
episode_rewards = []

for ep in range(episodes):
    observation, info = env.reset()
    done = False
    total_reward = 0
    episode_length = 0
    
    while not done:
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, _, info = env.step(action)
        total_reward += reward
        episode_length += 1
        
    episode_lengths.append(episode_length)
    episode_rewards.append(total_reward)
    
    print(f"Episode {ep + 1}: Length = {episode_length}, Reward = {total_reward}")

env.close()


print("Episode Lengths:", episode_lengths)
print("Episode Rewards:", episode_rewards)
