import gymnasium as gym
from stable_baselines3 import PPO
from custom_wrapper import RewardHackingWrapper

# 1. Création de l'environnement de base enveloppé avec le piège
base_env = gym.make("LunarLander-v3")
env = RewardHackingWrapper

# 2. Initialisation et entraînement du modèle PPO
model = PPO("MlpPolicy", env, verbose=1)

print("Début de l'entraînement avec Reward Hacking (bonus de sur-place)...")
model.learn(total_timesteps=400_000)

# 3. Sauvegarde de ce modèle "corrompu"
model.save("ppo_lunar_lander_hacked")
print("Modèle piégé sauvegardé sous 'ppo_lunar_lander_hacking.zip' !")