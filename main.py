import pickle
import random
import os
from collections import defaultdict
from ttt_train import TicTacToeEnv, QLearningAgent

env = TicTacToeEnv()
agent = QLearningAgent()

# Load existing Q-table if it exists
if os.path.exists("qtable.pkl"):
    with open("qtable.pkl", "rb") as f:
        agent.q = defaultdict(float, pickle.load(f))
    print("Saved bot loaded.")
else:
    print("No saved bot found. You should train before playing!")

def save():
    with open("qtable.pkl", "wb") as f:
        pickle.dump(dict(agent.q), f)
    print("Bot saved.")

def train(episodes):
    agent.epsilon = 1.0  # reset exploration for training
    print(f"Training for {episodes} episodes...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)
            next_state, reward, done = env.step(action, 1)

            if not done:
                opp_action = random.choice(env.available_actions())
                next_state, opp_reward, done = env.step(opp_action, -1)
                if done and opp_reward == -1:
                    reward = -1

            next_actions = env.available_actions()
            agent.update(state, action, reward, next_state, next_actions, done)
            state = next_state

        if (episode + 1) % (episodes // 10) == 0:  # progress every 10%
            print(f"  {episode + 1}/{episodes} episodes done...")

    print("Training complete!")
    save()

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: '.'}
    for i in range(0, 9, 3):
        print(f" {symbols[board[i]]} | {symbols[board[i+1]]} | {symbols[board[i+2]]}")
        if i < 6:
            print("---|---|---")
    print()

def play():
    agent.epsilon = 0  # bot plays its best
    state = env.reset()
    done = False

    print("\nBoard positions:")
    print(" 0 | 1 | 2 \n---|---|---\n 3 | 4 | 5 \n---|---|---\n 6 | 7 | 8 \n")
    print("You are O, bot is X.\n")

    while not done:
        print_board(env.board)

        available = env.available_actions()
        while True:
            try:
                move = int(input(f"Your move {available}: "))
                if move in available:
                    break
                print("Invalid move, pick an empty cell.")
            except ValueError:
                print("Please enter a number.")

        prev_state = state
        state, reward, done = env.step(move, -1)
        print_board(env.board)

        if done:
            print("You win!" if reward == -1 else "It's a draw!")
            agent.update(prev_state, move, reward, state, [], done)
            break

        actions = env.available_actions()
        bot_prev_state = state
        bot_move = agent.choose_action(state, actions)
        print(f"Bot plays: {bot_move}")
        state, reward, done = env.step(bot_move, 1)

        next_actions = env.available_actions()
        agent.update(bot_prev_state, bot_move, reward, state, next_actions, done)

        if done:
            print_board(env.board)
            print("Bot wins!" if reward == 1 else "It's a draw!")

    save()

# --- Main menu ---
while True:
    print("\n=== Tic-Tac-Toe ML Bot ===")
    print("1. Train the bot")
    print("2. Play against the bot")
    print("3. Quit")

    choice = input("\nChoose an option (1-3): ").strip()

    if choice == "1":
        while True:
            try:
                episodes = int(input("How many episodes to train? "))
                if episodes > 0:
                    break
                print("Please enter a number greater than 0.")
            except ValueError:
                print("Please enter a valid number.")
        train(episodes)

    elif choice == "2":
        play()

    elif choice == "3":
        print("Goodbye!")
        break

    else:
        print("Invalid option, please choose 1, 2, or 3.")