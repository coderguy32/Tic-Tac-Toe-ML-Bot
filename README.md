# Tic-Tac-Toe ML Bot

A tic-tac-toe bot that teaches itself how to play through Q-Learning. It starts out
making completely random moves and gradually figures out how to not lose through
trial and error over thousands of games.

## How it works

Every time the bot makes a move, it gets rewarded for winning and punished for losing.
It tracks all of this in a Q-table, which is basically a giant cheat sheet of
"in this situation, this move is a good/bad idea." After enough games it gets pretty
hard to beat.

## Files

- `ttt_train.py` — the game logic and the bot
- `main.py` — the menu you actually run

## Setup

Just needs Python, no external libraries required.
```
git clone <your repo url>
cd tic_tac_toe_ml
python main.py
```

## Usage

When you run `main.py` you get a simple menu:

- **Train the bot** — pick how many episodes to run. 100,000 is a good starting point
- **Play against it** — try to beat it in the terminal

The bot saves its progress automatically so it keeps getting better the more you train it.
try beating it after 500,000 episodes
