import gym
from stable_baselines3 import DQN

import os 


models_dir = "model/DQN"
logdir ="logs"

if not os.path.exists(models_dir): #bu dizinler var mi yok mu
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = gym.make("LunarLander-v2", render_mode = "human")

env.reset()

model = DQN("MlpPolicy", env, verbose =1,tensorboard_log=logdir) #mlp yapay sinir agi regresyon
TIMESTEPS = 10000
for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="DQN") #reset.adim sifirlanmayacak
    model.save(f"{models_dir}/{TIMESTEPS*i}")

      
env.close()

'''
rollout/

1.ep_len_mean: tamamlanan adimlarin ortalama uzunlugu
2.ep_rew_mean:ortalama odul sayisi

time/

1.fps: saniyede kac kez egitildi
2.iterations:toplam iterasyon sayisi
3.total_elapsed: toplam egitim suresi/sn
4.total_timesteps: zaman adimi(model cevre ile kac kere etkilesimde bulunmus)


train/

1.approx_kl: ppo kararliligi : dusuk deger daha iyi
2.clip_fraction: ppo clip guncellemelerinin kirpilmasi # politika butun olasiliklari alir kirpar
3.clip_range: clip mekanizmasinin araligiyla kirpilma belirlenir
4.entropy_loss:modelin degiliminin duzensizligi /duzensizlik kesfi arttirir
5.explained_variance: gercek degere ne kadar yakin? 1/iyi 0/eh
6.learning_rate: modelin ogrenme orani / modelin agirliklari ne kadar hizli guncellenir
7.loss: modelin kaybi / gerceklikten ne kadar uzak
8.policy_gradient_loss: politika gradyan kaybi / politika guncelleme
9.value_loss: kaybi olcer grcek ile ongorulenin farki












'''