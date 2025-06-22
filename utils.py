import numpy as np

def create_empty_board():
    return np.zeros((3, 3), dtype=int)

def get_valid_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == 0]

def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] != 0:
            return board[0][j]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    return 0

def is_board_full(board):
    return 0 not in board

def get_game_state(board):
    return tuple(board.flatten())

def count_winning_moves(board, player):
    count = 0
    for move in get_valid_moves(board):
        temp = board.copy()
        temp[move[0]][move[1]] = player
        if check_winner(temp) == player:
            count += 1
    return count

def is_fork(board, player):
    for move in get_valid_moves(board):
        temp = board.copy()
        temp[move[0]][move[1]] = player
        if count_winning_moves(temp, player) >= 2:
            return True
    return False

def get_reward(winner, player, moves_left=None, board=None, action=None):
    if winner == player:
        return 1.0 + (moves_left * 0.01 if moves_left is not None else 0)
    elif winner == 3 - player:
        return -1.0
    elif winner == 0:
        return 0.5
    return 0.0

def get_defensive_reward(board, player, action):
    if board is None or action is None:
        return 0.0
    opponent = 3 - player
    temp_board = board.copy()
    # Immediate block
    winning_move = None
    for i in range(3):
        for j in range(3):
            if temp_board[i][j] == 0:
                temp_board[i][j] = opponent
                if check_winner(temp_board) == opponent:
                    winning_move = (i, j)
                temp_board[i][j] = 0
    if winning_move and action == winning_move:
        return 1.5  # Immediate block
    # Fork block
    temp_board = board.copy()
    temp_board[action[0]][action[1]] = player
    if is_fork(temp_board, opponent):
        return 1.0  # Blocked a fork
    # Proactive: block two-in-a-row
    for move in get_valid_moves(temp_board):
        temp_board2 = temp_board.copy()
        temp_board2[move[0]][move[1]] = opponent
        if check_winner(temp_board2) == opponent:
            return 0.2  # Small reward for breaking up a threat
    return 0.0

def print_board(board):
    """Print the board in a readable format."""
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    for i in range(3):
        print(f" {symbols[board[i][0]]} | {symbols[board[i][1]]} | {symbols[board[i][2]]} ")
        if i < 2:
            print("-----------")
    print()

def get_all_symmetries(board, action):
    """
    Generate all symmetric (rotated/flipped) versions of the board and action.
    Returns a list of (sym_board, sym_action) tuples.
    """
    boards = []
    actions = []
    b = board.copy()
    a = action
    for k in range(4):  # 0, 90, 180, 270 degree rotations
        boards.append(np.rot90(b, k))
        actions.append(rotate_action(a, k))
    b = np.fliplr(board)
    for k in range(4):
        boards.append(np.rot90(b, k))
        actions.append(rotate_action(flip_action(a), k))
    return [(get_game_state(boards[i]), actions[i]) for i in range(8)]

def rotate_action(action, k):
    """Rotate action (i, j) k times 90 degrees counterclockwise on a 3x3 board."""
    i, j = action
    for _ in range(k):
        i, j = j, 2 - i
    return (i, j)

def flip_action(action):
    """Flip action (i, j) horizontally on a 3x3 board."""
    i, j = action
    return (i, 2 - j)
