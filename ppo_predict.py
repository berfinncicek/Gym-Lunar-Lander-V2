import gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2", render_mode="human")

model_path = "model/PPO-1710841779/20000.zip"
model = PPO.load(model_path, env=env)

episodes = 10

for ep in range(episodes):
    observation, info = env.reset(), {}
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        action, _ = model.predict(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    
    print("Episode:", ep + 1, "Total Reward:", total_reward)

# Tüm episode'lar tamamlandıktan sonra ortamı kapat
env.close()
