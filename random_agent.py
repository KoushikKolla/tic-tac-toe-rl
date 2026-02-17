import random

class RandomAgent:
    def __init__(self, name="RandomAgent"):
        self.name = name

    def select_action(self, state, available_actions):
        return random.choice(available_actions)
