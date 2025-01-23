#### 1
>---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[5], line 14
     11 action_noise = NormalActionNoise(mean = np.zeros(env.action_space.shape), sigma = 0.3 * np.ones(env.action_space.shape))
     12 model = DDPG('MlpPolicy', env, policy_kwargs = dict(net_arch=[256, 128]), action_noise = action_noise, verbose = 1)
---> 14 model.learn(total_timesteps = 100000)
     16 model.save("ddpg_cartpole_model")
     17 # 评估训练后的模型
File d:\Study\anaconda\envs\new\lib\site-packages\stable_baselines3\ddpg\ddpg.py:123, in DDPG.learn(self, total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
    114 def learn(
    115     self: SelfDDPG,
    116     total_timesteps: int,
   (...)
    121     progress_bar: bool = False,
    122 ) -> SelfDDPG:
--> 123     return super().learn(
    124         total_timesteps=total_timesteps,
    125         callback=callback,
    126         log_interval=log_interval,
    127         tb_log_name=tb_log_name,
    128         reset_num_timesteps=reset_num_timesteps,
    129         progress_bar=progress_bar,
    130     )
...
    358 elif self.steps_beyond_terminated is None:
    359     # Pole just fell!
    360     self.steps_beyond_terminated = 0
TypeError: 'numpy.float64' object is not callable


cause: 

    reward += 10 * self.reward() instead of reward += 10 * reward() 



容易产生局部最优

改撞墙的惩罚为reward = -200
减小稳定角度cos范围