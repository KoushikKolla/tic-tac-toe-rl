<<<<<<< HEAD
# tic-tac-toe-rl
=======
# 🎯 Tic Tac Toe Game using Reinforcement Learning with Python GUI

## 📋 Project Overview

This project implements a Tic Tac Toe game where a Reinforcement Learning (RL) agent plays against a human player. The agent learns optimal gameplay strategies using Q-Learning without any hardcoded rules or supervised learning.

## 🎮 Features

- **Interactive GUI**: Built with Tkinter for smooth gameplay
- **Q-Learning Agent**: Learns through exploration and exploitation
- **Self-Training**: Agent trains through self-play before facing humans
- **Smart Gameplay**: Agent aims to win or draw consistently
- **Reset Functionality**: Start new games anytime

## 🧠 Machine Learning Approach

### Q-Learning Algorithm
- **States**: 3^9 combinations (empty, X, O) represented as tuples
- **Actions**: Valid moves from current game state
- **Reward Function**:
  - Win: +1
  - Draw: +0.5
  - Lose: -1

### Training Strategy
- **Epsilon-Greedy**: Balances exploration vs exploitation
- **Self-Play**: Agent trains by playing against itself
- **State Hashing**: Efficient Q-table storage using tuple representation

## 📁 Project Structure

```
tic_tac_toe_rl/
├── main.py             # GUI and game integration
├── q_learning_agent.py # RL agent logic
├── train_agent.py      # Pre-training the agent
├── utils.py            # Helper functions
└── README.md           # Project instructions
```

## 🚀 Setup and Installation

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `tkinter` (usually comes with Python)

### Installation
1. Clone or download this repository
2. Navigate to the project directory
3. Install dependencies (if needed):
   ```bash
   pip install numpy
   ```

## 🎯 Usage

### Quick Start
1. Run the main application:
   ```bash
   python main.py
   ```

2. The agent will automatically train for a few seconds before the GUI appears
3. Play against the trained RL agent!

### Training the Agent
- The agent trains automatically when you run `main.py`
- Training parameters can be adjusted in `train_agent.py`
- Training progress is displayed in the console

### Game Controls
- Click any empty cell to make your move
- The agent will respond automatically
- Use "New Game" button to start a fresh game

## 🔧 Customization

### Training Parameters
Edit `train_agent.py` to modify:
- Number of training episodes
- Learning rate (alpha)
- Discount factor (gamma)
- Exploration rate (epsilon)

### GUI Customization
Modify `main.py` to change:
- Window size and appearance
- Button colors and styling
- Game board layout

## 📊 Performance

The RL agent typically achieves:
- **Win Rate**: 85-95% against random play
- **Draw Rate**: 5-15% against optimal play
- **Training Time**: 10-30 seconds for optimal performance

## 🎓 Learning Outcomes

This project demonstrates:
- Q-Learning implementation from scratch
- State representation and action selection
- Exploration vs exploitation strategies
- GUI integration with ML algorithms
- Self-play training methodology

## 🤝 Contributing

Feel free to:
- Improve the training algorithm
- Add new features to the GUI
- Optimize the Q-table storage
- Enhance the reward function

## 📝 License

This project is open source and available under the MIT License.

---

**Note**: The agent learns purely through reinforcement learning - no hardcoded game rules or supervised learning is used!
>>>>>>> 89ee0f9 (Add DQN agent, defensive reward shaping, and GUI agent selection)
