import random

class WeakTeacherAgent:
    def __init__(self, name="WeakTeacherAgent"):
        self.name = name

    def select_action(self, state, available_actions, player_mark, opponent_mark=None):
        # Try to win
        for action in available_actions:
            next_state = list(state)
            next_state[action] = player_mark
            if self.check_win(next_state, player_mark):
                return action
        # Try to block opponent's win
        if opponent_mark is not None:
            for action in available_actions:
                next_state = list(state)
                next_state[action] = opponent_mark
                if self.check_win(next_state, opponent_mark):
                    return action
        # Otherwise, pick random
        return random.choice(available_actions)

    def check_win(self, board, mark):
        # Board is a flat list of 9 elements
        win_states = [
            [0,1,2], [3,4,5], [6,7,8], # rows
            [0,3,6], [1,4,7], [2,5,8], # cols
            [0,4,8], [2,4,6]           # diags
        ]
        for indices in win_states:
            if all(board[i] == mark for i in indices):
                return True
        return False 