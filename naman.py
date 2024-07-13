# This is my first program for Tic Tac Toe AI


# Same structure
board = ["-", "-", "-", 
         "-", "-", "-", 
         "-", "-", "-"]

# Print the game structure
def print_board():
    print("\n")
    print(board[0] + " | " + board[1] + " | " + board[2])
    print(board[3] + " | " + board[4] + " | " + board[5])
    print(board[6] + " | " + board[7] + " | " + board[8])
    print("\n")

# Player's turn
def take_turn(player):
    if player == "X":
        while True:
            try:
                position = input("Choose a position from 1-9: ")
                if position in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    position = int(position) - 1
                    if board[position] == "-":
                        board[position] = player
                        break
                    else:
                        print("Position already taken. Choose a different position.")
                else:
                    print("Invalid input. Choose a position from 1-9.")
            except ValueError:
                print("Invalid input. Please enter a number from 1-9.")
    else:
        position = best_move(board, player)
        board[position] = player
        print(f"AI (O) chooses position {position + 1}")
    print_board()

# If game is over
def check_game_over():
    # Define the winning combinations
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8), # Horizontal
        (0, 3, 6), (1, 4, 7), (2, 5, 8), # Vertical
        (0, 4, 8), (2, 4, 6)             # Diagonal
    ]
    
    # If win
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] != "-":
            return "win"
    
    # If tie
    if "-" not in board:
        return "tie"
    
    # If not over
    return "play"

# Minimax algorithm with Alpha-Beta Pruning for the AI
def minimax(board, player, alpha, beta):
    opponent = "O" if player == "X" else "X"
    game_result = check_game_over()

    if game_result == "win":
        return {"score": 1} if player == "O" else {"score": -1}
    elif game_result == "tie":
        return {"score": 0}

    best_score = -float('inf') if player == "O" else float('inf')
    best_move = None

    for i in range(9):
        if board[i] == "-":
            board[i] = player
            result = minimax(board, opponent, alpha, beta)
            board[i] = "-"
            score = result["score"]

            if player == "O":
                if score > best_score:
                    best_score = score
                    best_move = i
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = i
                beta = min(beta, score)

            if alpha >= beta:
                break

    return {"position": best_move, "score": best_score}

# Best move for the AI
def best_move(board, player):
    return minimax(board, player, -float('inf'), float('inf'))["position"]

# Main game loop
def play_game():
    print_board()
    current_player = "X"
    while True:
        take_turn(current_player)
        game_result = check_game_over()
        if game_result == "win":
            print(f"{current_player} wins!")
            break
        elif game_result == "tie":
            print("It's a tie!")
            break
        else:
            # Switch to the other player
            current_player = "O" if current_player == "X" else "X"
# Start the game
play_game()
