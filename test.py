import gymnasium as gym
from stable_baselines3 import PPO

# Charger l'environnement
env = gym.make("LunarLander-v3", render_mode="human")

# Charger le modèle que vous avez ramené de Colab
model = PPO.load("ppo_lunar_lander")

# Lancer quelques parties pour voir le résultat
for episode in range(3):
    observation, info = env.reset()
    done = False
    score = 0
    
    while not done:
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        done = terminated or truncated
        
    print(f"Essai {episode + 1} - Score : {score:.2f}")

env.close()