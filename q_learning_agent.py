import numpy as np
import random
from collections import deque
from utils import get_valid_moves, check_winner, is_board_full, get_game_state, get_reward, get_all_symmetries, get_defensive_reward

class QLearningAgent:
    def __init__(self, player, learning_rate=0.1, discount_factor=0.9, epsilon=0.4, epsilon_decay=0.9999, epsilon_min=0.01):
        self.player = player
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.training_episodes = 0
        self.memory = deque(maxlen=50000)
        self.hindsight_memory = deque(maxlen=50000)
    
    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)
    
    def set_q_value(self, state, action, value):
        board = np.array(state).reshape(3, 3)
        for sym_state, sym_action in get_all_symmetries(board, action):
            self.q_table[(sym_state, sym_action)] = value
    
    def choose_action(self, board, training=True):
        valid_moves = get_valid_moves(board)
        if not valid_moves: return None
        state = get_game_state(board)
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        else:
            return self.get_best_action(state, valid_moves)
    
    def get_best_action(self, state, valid_moves):
        if not valid_moves: return None
        q_values = [(action, self.get_q_value(state, action)) for action in valid_moves]
        max_q = max(q_values, key=lambda x: x[1])[1]
        best_actions = [action for action, q_val in q_values if q_val == max_q]
        return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, next_valid_moves):
        current_q = self.get_q_value(state, action)
        if next_valid_moves:
            max_next_q = max(self.get_q_value(next_state, next_action) for next_action in next_valid_moves)
        else:
            max_next_q = 0
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.set_q_value(state, action, new_q)
    
    def remember(self, state, action, reward, next_state, next_valid_moves, done):
        self.memory.append((state, action, reward, next_state, next_valid_moves, done))
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, next_valid_moves, done in batch:
            self.update_q_value(state, action, reward, next_state, next_valid_moves)
        if len(self.hindsight_memory) > batch_size:
            hindsight_batch = random.sample(self.hindsight_memory, batch_size)
            for state, action, _, _, _, _ in hindsight_batch:
                self.update_q_value(state, action, 0.5, state, [])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, teacher_agent=None):
        board = np.zeros((3, 3), dtype=int)
        current_player = 1
        opponent_agent = teacher_agent if teacher_agent else QLearningAgent(3 - self.player, epsilon=self.epsilon)
        
        episode_memory = []
        
        while True:
            done = False
            if current_player == self.player:
                action = self.choose_action(board, training=True)
                if action is None: break
                
                current_state = get_game_state(board)
                defensive_reward = get_defensive_reward(board, self.player, action)
                
                board[action[0]][action[1]] = current_player
                next_state = get_game_state(board)
                
                winner = check_winner(board)
                if winner != 0 or is_board_full(board): done = True
                
                reward = get_reward(winner, self.player, np.count_nonzero(board==0)) + defensive_reward
                next_valid_moves = get_valid_moves(board)
                
                episode_memory.append((current_state, action, reward, next_state, next_valid_moves, done))
                
                if done: break
            else:
                action = opponent_agent.choose_action(board, training=True)
                if action is None: break
                board[action[0]][action[1]] = current_player
                if check_winner(board) != 0 or is_board_full(board): break
            
            current_player = 3 - current_player
            
        for experience in episode_memory:
            self.remember(*experience)
            self.hindsight_memory.append(experience)
            
        self.decay_epsilon()
        self.training_episodes += 1
        self.replay()
        return check_winner(board)
    
    def get_action_for_gui(self, board):
        return self.choose_action(board, training=False)
    
    def save_q_table(self, filename):
        import pickle
        with open(filename, 'wb') as f: pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename):
        import pickle
        with open(filename, 'rb') as f: self.q_table = pickle.load(f)

class TeacherAgent:
    def choose_action(self, board, training=True):
        valid_moves = get_valid_moves(board)
        if not valid_moves: return None
        
        # 1. Win if possible
        for move in valid_moves:
            temp_board = board.copy()
            temp_board[move[0]][move[1]] = 2
            if check_winner(temp_board) == 2: return move
        
        # 2. Block if necessary
        for move in valid_moves:
            temp_board = board.copy()
            temp_board[move[0]][move[1]] = 1
            if check_winner(temp_board) == 1:
                return move
        
        # 3. Create a fork
        for move in valid_moves:
            temp_board = board.copy()
            temp_board[move[0]][move[1]] = 2
            winning_moves = 0
            for next_move in get_valid_moves(temp_board):
                next_board = temp_board.copy()
                next_board[next_move[0]][next_move[1]] = 2
                if check_winner(next_board) == 2:
                    winning_moves += 1
            if winning_moves >= 2:
                return move

        # 4. Block opponent's fork
        for move in valid_moves:
            # Temporarily make a player 1 move and see if it creates a fork
            temp_board = board.copy()
            temp_board[move[0]][move[1]] = 1 # Opponent's potential move
            
            # Count opponent's winning opportunities after this move
            opponent_wins = 0
            possible_next_moves = get_valid_moves(temp_board)
            for next_move in possible_next_moves:
                check_board = temp_board.copy()
                check_board[next_move[0]][next_move[1]] = 1
                if check_winner(check_board) == 1:
                    opponent_wins += 1
            if opponent_wins >= 2:
                # If opponent can create a fork, we must block that move
                return move

        # 5. Center
        if (1, 1) in valid_moves: return (1, 1)
        
        # 6. Opposite Corner
        corners = [(0,0), (0,2), (2,0), (2,2)]
        for i in range(len(corners)):
            if board[corners[i][0]][corners[i][1]] == 1:
                opposite_corner = corners[len(corners) - 1 - i]
                if board[opposite_corner[0]][opposite_corner[1]] == 0:
                    return opposite_corner

        # 7. Empty Corner
        for move in corners:
            if move in valid_moves: return move
        
        # 8. Empty Side
        sides = [(0,1), (1,0), (1,2), (2,1)]
        for move in sides:
            if move in valid_moves: return move
            
        return random.choice(valid_moves)
