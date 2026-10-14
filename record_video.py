import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# 1. Création de l'environnement avec enregistrement vidéo activé
# (On peut utiliser le modèle normal ou le modèle "hacked")
env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="./lunar_lander_video", episode_trigger=lambda e: True)

# 2. Charger le modèle de votre choix (ex: le modèle piégé ou le modèle de base)
model = PPO.load("ppo_lunar_lander_hacked")

# 3. Lancer un essai pour enregistrer la vidéo (.mp4)
observation, info = env.reset()
done = False

while not done:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
print("Vidéo enregistrée dans le dossier './lunar_lander_video' !")