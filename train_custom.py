import gymnasium as gym
from stable_baselines3 import PPO
from custom_wrapper import TimePnaltyWrapper

# 1. Création de l'environnement de base
base_env = gym.make("LunarLander-v3")

# 2. On enveloppe notre environnement avec notre règle personnalisé (ex: -0.1 point par frame)
env = TimePnaltyWrapper(base_env, penalty_amount=0.1)

# 3. Initialisation et entraînement de l'agent PPO
model = PPO("MlpPolicy", env, verbose=1)

print("Début de l'entraînement avec Reward Shaping (pénalité de temps)...")
model.learn(total_timesteps=500_000) # 500k pas suffisant souvent pour adapter le modèle

# 4. Sauvegarde de ce modèle
model.save("ppo_lunar_lander_custom")
print("Modèle personnalisé entraîné et sauvegardé sous 'ppo_lunar_lander_custom.zip' !")