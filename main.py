import tkinter as tk
from tkinter import messagebox
import numpy as np
import threading
import time
from q_learning_agent import QLearningAgent
from train_agent import train_agent, train_defensive_scenarios
from utils import check_winner, is_board_full, is_fork, get_defensive_reward
from random_agent import RandomAgent
from weak_teacher_agent import WeakTeacherAgent
from strong_teacher_agent import StrongTeacherAgent
from dqn_agent import DQNAgent

class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('Tic Tac Toe - Reinforcement Learning Agent')
        self.root.geometry('500x600')
        self.root.configure(bg='#2c3e50')
        
        self.board = np.zeros((3, 3), dtype=int)
        self.game_active = False
        self.agent = None
        
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        self.status_label = None
        self.new_game_btn = None
        self.progress_label = None
        
        self.setup_gui()
        self.train_agent_async()
    
    def setup_gui(self):
        title_label = tk.Label(
            self.root,
            text='Tic Tac Toe vs RL Agent',
            font=('Arial', 20, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        self.status_label = tk.Label(
            self.root,
            text='Training agent with defensive focus... Please wait.',
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        self.status_label.pack(pady=10)
        
        self.progress_label = tk.Label(
            self.root,
            text='Initializing...',
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#bdc3c7'
        )
        self.progress_label.pack(pady=5)
        
        board_frame = tk.Frame(self.root, bg='#2c3e50')
        board_frame.pack(pady=20)
        
        for i in range(3):
            for j in range(3):
                button = tk.Button(
                    board_frame,
                    text='',
                    font=('Arial', 16, 'bold'),
                    width=6,
                    height=3,
                    bg='#34495e',
                    fg='white',
                    command=lambda row=i, col=j: self.make_move(row, col),
                    state='disabled'
                )
                button.grid(row=i, column=j, padx=5, pady=5)
                self.buttons[i][j] = button
        
        self.new_game_btn = tk.Button(
            self.root,
            text='New Game',
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            command=self.new_game,
            state='disabled'
        )
        self.new_game_btn.pack(pady=20)
        
        # Info text
        info_text = """
🤖 Enhanced RL Agent Features:
• Q-Learning with epsilon decay
• Symmetry learning (8x faster)
• Experience replay for stability
• Defensive reward shaping
• 15,000 training episodes
• Specialized defensive training

🎮 How to Play:
• You are X (goes first)
• Agent is O (trained to block and win)
• Click any empty cell to make your move
• Agent will block your winning moves!
        """
        
        info_label = tk.Label(
            self.root,
            text=info_text,
            font=('Arial', 9),
            bg='#2c3e50',
            fg='#bdc3c7',
            justify=tk.LEFT
        )
        info_label.pack(pady=10)
    
    def train_agent_async(self):
        def train():
            # Train with improved parameters and defensive focus
            self.agent = train_agent(
                episodes=15000,  # More episodes for better learning
                learning_rate=0.1,
                discount_factor=0.9,
                epsilon=0.3,  # Higher initial exploration
                epsilon_decay=0.9995,  # Gradual decay
                epsilon_min=0.01  # Minimum exploration
            )
            
            # Additional defensive training
            train_defensive_scenarios(self.agent, num_scenarios=2000)
            
            self.root.after(0, self.on_training_complete)
        
        training_thread = threading.Thread(target=train)
        training_thread.daemon = True
        training_thread.start()
    
    def on_training_complete(self):
        self.progress_label.pack_forget()
        self.status_label.config(
            text='Agent trained! You are X, Agent is O. Your turn!',
            fg='#27ae60'
        )
        self.new_game_btn.config(state='normal')
        self.game_active = True
        self.enable_buttons()
    
    def make_move(self, row, col):
        if not self.game_active or self.board[row][col] != 0:
            return
        
        self.board[row][col] = 1
        self.buttons[row][col].config(text='X', bg='#3498db')
        
        if check_winner(self.board) == 1:
            self.end_game('Congratulations! You won! 🎉')
            return
        
        if is_board_full(self.board):
            self.end_game('It\'s a draw! 🤝')
            return
        
        self.status_label.config(text='Agent is thinking...', fg='#f39c12')
        self.root.update()
        
        # Add small delay to make agent thinking visible
        time.sleep(0.5)
        
        ai_action = self.agent.get_action_for_gui(self.board)
        
        if ai_action:
            self.board[ai_action[0]][ai_action[1]] = 2
            self.buttons[ai_action[0]][ai_action[1]].config(text='O', bg='#e74c3c')
            
            if check_winner(self.board) == 2:
                self.end_game('Agent wins! 🤖')
                return
            
            if is_board_full(self.board):
                self.end_game('It\'s a draw! 🤝')
                return
        
        self.status_label.config(text='Your turn! You are X', fg='#27ae60')
    
    def end_game(self, message):
        self.game_active = False
        self.disable_buttons()
        self.status_label.config(text=message, fg='#e74c3c')
        
        result = messagebox.askyesno('Game Over', f'{message}\n\nWould you like to play again?')
        if result:
            self.new_game()
    
    def new_game(self):
        self.board = np.zeros((3, 3), dtype=int)
        
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='', bg='#34495e')
        
        self.game_active = True
        self.status_label.config(text='New game! You are X, Agent is O. Your turn!', fg='#27ae60')
        self.enable_buttons()
    
    def enable_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(state='normal')
    
    def disable_buttons(self):
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(state='disabled')

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
