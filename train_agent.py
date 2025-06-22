import numpy as np
import time
from q_learning_agent import QLearningAgent, TeacherAgent
from utils import print_board, check_winner

def train_agent(episodes=50000, learning_rate=0.1, discount_factor=0.9, epsilon=0.5, epsilon_decay=0.99995, epsilon_min=0.01):
    """
    Train the Q-Learning agent with a teacher agent to improve defensive play.
    """
    print("🤖 Training Q-Learning Agent with an Expert Teacher...")
    print(f"📊 Episodes: {episodes}")
    print(f"📈 Learning Rate: {learning_rate}")
    print(f"🎯 Discount Factor: {discount_factor}")
    print(f"🔍 Initial Epsilon: {epsilon}")
    print(f"📉 Epsilon Decay: {epsilon_decay}")
    print(f"🔽 Min Epsilon: {epsilon_min}")
    print("-" * 60)
    
    agent = QLearningAgent(
        player=1,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )
    
    teacher = TeacherAgent()
    
    wins, draws, losses = 0, 0, 0
    start_time = time.time()
    
    for episode in range(episodes):
        if episode < 40000: # 80% of training with teacher
            winner = agent.train_episode(teacher_agent=teacher)
        else: # Final 20% self-play to generalize
            winner = agent.train_episode()
            
        if winner == 1: wins += 1
        elif winner == 2: losses += 1
        else: draws += 1
        
        if (episode + 1) % 5000 == 0:
            win_rate = wins / (episode + 1) * 100
            draw_rate = draws / (episode + 1) * 100
            loss_rate = losses / (episode + 1) * 100
            print(f"Episode {episode + 1:5d} | "
                  f"W: {win_rate:5.1f}% | "
                  f"D: {draw_rate:5.1f}% | "
                  f"L: {loss_rate:5.1f}% | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Q-table: {len(agent.q_table)}")
    
    training_time = time.time() - start_time
    print("-" * 60)
    print("✅ Training Complete!")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    print(f"📊 Final Stats: W: {wins}, D: {draws}, L: {losses}")
    print(f"🧠 Q-table entries: {len(agent.q_table)}")
    
    return agent

def train_defensive_scenarios(agent, num_scenarios=1000):
    """
    Train the agent specifically on defensive scenarios.
    Creates board states where the agent needs to block.
    """
    print(f"\n🛡️ Training on Defensive Scenarios ({num_scenarios} scenarios)...")
    
    from utils import check_winner, get_defensive_reward
    
    successful_blocks = 0
    
    for scenario in range(num_scenarios):
        # Create specific blocking scenarios
        if scenario < num_scenarios // 3:
            # Horizontal blocking scenarios
            board = np.zeros((3, 3), dtype=int)
            row = np.random.randint(0, 3)
            col1, col2 = np.random.choice([0, 1, 2], size=2, replace=False)
            board[row][col1] = 2  # Opponent piece
            board[row][col2] = 2  # Opponent piece
            expected_block = (row, 3 - col1 - col2)
            
        elif scenario < 2 * num_scenarios // 3:
            # Vertical blocking scenarios
            board = np.zeros((3, 3), dtype=int)
            col = np.random.randint(0, 3)
            row1, row2 = np.random.choice([0, 1, 2], size=2, replace=False)
            board[row1][col] = 2  # Opponent piece
            board[row2][col] = 2  # Opponent piece
            expected_block = (3 - row1 - row2, col)
            
        else:
            # Diagonal blocking scenarios
            board = np.zeros((3, 3), dtype=int)
            if np.random.random() < 0.5:
                # Main diagonal
                board[0][0] = 2
                board[1][1] = 2
                expected_block = (2, 2)
            else:
                # Anti-diagonal
                board[0][2] = 2
                board[1][1] = 2
                expected_block = (2, 0)
        
        # Place some agent pieces randomly
        agent_positions = np.random.choice(9, size=np.random.randint(1, 3), replace=False)
        for pos in agent_positions:
            i, j = pos // 3, pos % 3
            if board[i][j] == 0:  # Only if empty
                board[i][j] = 1  # Agent piece
        
        # Let agent make a move
        action = agent.choose_action(board, training=True)
        if action:
            # Check if it's a good defensive move
            defensive_reward = get_defensive_reward(board, 1, action)
            
            # Give extra reward if it blocks correctly
            if action == expected_block:
                defensive_reward += 0.3
                successful_blocks += 1
            
            # Update Q-value for this defensive scenario
            current_state = tuple(board.flatten())
            agent.update_q_value(current_state, action, defensive_reward, current_state, [])
    
    print(f"✅ Defensive scenario training complete!")
    print(f"📊 Successful blocks: {successful_blocks}/{num_scenarios} ({successful_blocks/num_scenarios*100:.1f}%)")

def test_agent(agent, test_games=100):
    print(f"\n🧪 Testing agent against random player ({test_games} games)...")
    wins, draws, losses = 0, 0, 0
    
    for game in range(test_games):
        board = np.zeros((3, 3), dtype=int)
        current_player = 1
        
        while True:
            if current_player == 1:
                action = agent.get_action_for_gui(board)
                if action is None: break
                board[action[0]][action[1]] = current_player
            else:
                valid_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]
                if not valid_moves: break
                action = np.random.choice(len(valid_moves))
                move = valid_moves[action]
                board[move[0]][move[1]] = current_player
            
            winner = 0
            for i in range(3):
                if board[i][0] == board[i][1] == board[i][2] != 0: winner = board[i][0]
                if board[0][i] == board[1][i] == board[2][i] != 0: winner = board[0][i]
            if board[0][0] == board[1][1] == board[2][2] != 0: winner = board[0][0]
            if board[0][2] == board[1][1] == board[2][0] != 0: winner = board[0][2]
            
            if winner != 0 or 0 not in board: break
            
            current_player = 3 - current_player
            
        if winner == 1: wins += 1
        elif winner == 2: losses += 1
        else: draws += 1
        
    print(f"📊 Test Results: Wins: {wins}, Draws: {draws}, Losses: {losses}")
    return wins, draws, losses

def test_blocking_after_full_training(agent):
    print("\n" + "="*50)
    print("🛡️  Testing Fully Trained Agent's Blocking Behavior 🛡️")
    print("="*50)

    # Scenarios where player 2 (human) has a winning threat
    # Agent is player 1 (X)
    scenarios = [
        {'name': 'Block Horizontal', 'board': np.array([[0, 2, 2], [1, 0, 0], [1, 0, 0]]), 'expected_block': (0, 0)},
        {'name': 'Block Vertical', 'board': np.array([[2, 1, 0], [2, 1, 0], [0, 0, 0]]), 'expected_block': (2, 0)},
        {'name': 'Block Diagonal', 'board': np.array([[2, 1, 0], [0, 2, 0], [0, 1, 0]]), 'expected_block': (2, 0)},
        {'name': 'Block Anti-Diagonal', 'board': np.array([[1, 1, 2], [0, 2, 0], [0, 0, 0]]), 'expected_block': (2, 0)},
        {'name': 'Block Fork', 'board': np.array([[2, 0, 0], [0, 1, 0], [0, 0, 2]]), 'expected_block': (0, 1)},
    ]

    correct_blocks = 0
    total_scenarios = len(scenarios)

    # The agent was trained as player 1, so we test it as player 1
    original_player = agent.player
    agent.player = 1 

    for scenario in scenarios:
        print(f"\n--- Scenario: {scenario['name']} ---")
        board = scenario['board']
        print("Board state:")
        print_board(board)

        action = agent.get_action_for_gui(board)
        
        print(f"Agent chose to play at: {action}")
        print(f"Expected blocking move: {scenario['expected_block']}")

        if action == scenario['expected_block']:
            correct_blocks += 1
            print("✅ Agent blocked correctly!")
        else:
            print("❌ Agent failed to block.")
    
    agent.player = original_player # Restore original player

    print("\n" + "="*50)
    blocking_score = (correct_blocks / total_scenarios) * 100
    print(f"🛡️  Final Blocking Score: {blocking_score:.1f}% ({correct_blocks}/{total_scenarios})")
    if blocking_score > 80:
        print("🎉 Agent has learned excellent defensive strategies!")
    else:
        print("⚠️ Agent could still improve its defensive play.")

if __name__ == "__main__":
    # Train the agent with the expert teacher
    agent = train_agent()
    
    # Run the final blocking exam
    test_blocking_after_full_training(agent)
    
    # Save the fully trained agent
    agent.save_q_table("trained_agent_final.pkl")
    print("\n💾 Trained agent saved to 'trained_agent_final.pkl'")
