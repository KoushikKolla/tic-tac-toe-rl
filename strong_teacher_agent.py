import random

class StrongTeacherAgent:
    def __init__(self, name="StrongTeacherAgent"):
        self.name = name

    def select_action(self, state, available_actions, player_mark, opponent_mark=None):
        # Use minimax to choose the best move
        best_score = -float('inf')
        best_actions = []
        for action in available_actions:
            next_state = list(state)
            next_state[action] = player_mark
            score = self.minimax(next_state, False, player_mark, opponent_mark)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
        return random.choice(best_actions)

    def minimax(self, board, is_maximizing, player_mark, opponent_mark):
        winner = self.check_winner(board)
        if winner == player_mark:
            return 1
        elif winner == opponent_mark:
            return -1
        elif all(cell != 0 for cell in board):
            return 0

        available_actions = [i for i, cell in enumerate(board) if cell == 0]
        if is_maximizing:
            best_score = -float('inf')
            for action in available_actions:
                board[action] = player_mark
                score = self.minimax(board, False, player_mark, opponent_mark)
                board[action] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for action in available_actions:
                board[action] = opponent_mark
                score = self.minimax(board, True, player_mark, opponent_mark)
                board[action] = 0
                best_score = min(score, best_score)
            return best_score

    def check_winner(self, board):
        win_states = [
            [0,1,2], [3,4,5], [6,7,8], # rows
            [0,3,6], [1,4,7], [2,5,8], # cols
            [0,4,8], [2,4,6]           # diags
        ]
        for indices in win_states:
            if board[indices[0]] != 0 and all(board[i] == board[indices[0]] for i in indices):
                return board[indices[0]]
        return None 