🚀 Lunar Lander - Reinforcement Learning & Reward Shaping
Projet d'intelligence artificielle implémentant l'apprentissage par renforcement (Reinforcement Learning) pour résoudre l'environnement Lunar Lander via l'algorithme PPO (Proximal Policy Optimization), avec une exploration avancée du Reward Shaping et du Reward Hacking.

📌 Aperçu du Projet
Ce projet a pour objectif d'entraîner un agent autonome à faire atterrir une fusée entre deux drapeaux sans s'écraser. Il explore non seulement l'optimisation des politiques d'apprentissage via Stable-Baselines3 et Gymnasium, mais aussi l'impact critique de la conception des fonctions de récompense (Reward Shaping et mise en évidence de comportements de triche par Reward Hacking).

Aperçu de l'agent en action :
(Glissez-déposez votre fichier GIF ici une fois généré)

🛠️ Technologies Utilisées
Langage : Python 3.10+

Framework RL : Stable-Baselines3 (PPO)

Environnement Physique : Gymnasium (box2d)

Bibliothèques annexes : PyTorch, Pygame, NumPy

📂 Structure du Projet
Plaintext
📦 lunar-lander-rl
│
├── 📄 custom_wrapper.py        # Définition des wrappers personnalisés (Pénalité de temps & Reward Hacking)
├── 📄 train_custom.py          # Script d'entraînement avec le wrapper de pénalité de temps
├── 📄 train_hacking.py         # Script d'entraînement démontrant le Reward Hacking
├── 📄 test.py                  # Script de visualisation et d'évaluation des modèles
└── 📄 README.md                # Documentation du projet
⚙️ Implémentation du Reward Shaping (Extraits)
Le projet intègre des surcouches (wrappers) personnalisées pour modifier le comportement de l'agent en temps réel :

1. Pénalité de temps (custom_wrapper.py)
Python
import gymnasium as gym

class TimePenaltyWrapper(gym.Wrapper):
    """Ajoute une pénalité de temps pour forcer l'agent à atterrir rapidement."""
    def __init__(self, env, penalty_amount=0.1):
        super().__init__(env)
        self.penalty_amount = penalty_amount

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.penalty_amount
        return observation, reward, terminated, truncated, info
2. Démonstration de Reward Hacking
En sur-rémunérant la stabilité en vol (+1.0 par frame si la fusée est droite et lente), l'agent apprend à exploiter la faille en faisant du sur-place infini pour accumuler des points au détriment de la mission principale (l'atterrissage).

🚀 Installation et Utilisation en Local
Cloner le dépôt :

Bash
git clone https://github.com/votre-pseudo/lunar-lander-rl.git
cd lunar-lander-rl
Créer et activer un environnement virtuel :

Bash
python -m venv venv
# Sur Windows :
venv\Scripts\activate
# Sur macOS / Linux :
source venv/bin/activate
Installer les dépendances :

Bash
pip install "gymnasium[box2d]" stable-baselines3
Lancer la visualisation d'un modèle :

Bash
python test.py