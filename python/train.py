from stable_baselines3 import PPO
from env_wrapper import GameEnv

env = GameEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    gamma=0.99
)

model.learn(total_timesteps=5_000_000)

model.save("policy")