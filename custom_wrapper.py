import gymnasium as gym

class TimePenaltyWrapper(gym.Wrapper):
    """
    Un wrapper personnalisé qui ajoute une petite pénalité de temps à chaque frame 
    pour forcer l'agent à atterrir rapidement.
    """
    def __init__(self, env, penalty_amount=0.1):
        super().__init__(env)
        self.penalty_amount = penalty_amount

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # On applique la pénalité de temps sur la récompense
        reward -= self.penalty_amount
        return observation, reward, terminated, truncated, info

class RewardHackingWrapper(gym.Wrapper):
    """
    Wrapper de Reward Hacking : On donne un bonus de +1 par frame
    si la fusée est globalement droite et stable.
    Résultat : l'IA apprend à faire du sur-place pour farmer des points !
    """
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # observation[4] = inclinaison (tilt)
        # observation[2] & [3] = vitesses horizontale et verticale
        tilt = abs(observation[4])
        speed = abs(observation[2]) + abs(observation[3])

        # Si la fusée est stable et presque droite, on la sur-rémunère
        if tilt < 0.1 and speed < 0.5:
            reward += 1.0

        return observation, reward, terminated, truncated, info
        