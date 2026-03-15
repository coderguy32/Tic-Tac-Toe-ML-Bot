import random
from collections import defaultdict


class TicTacToeEnv:
    def reset(self):
        self.board = [0] * 9
        return tuple(self.board)
    
    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]  # fixed
    
    def check_winner(self):  # added
        wins = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        return None
    
    def step(self, action, player):
        self.board[action] = player
        winner = self.check_winner()
        done = winner is not None or not self.available_actions()
        reward = winner if winner else 0
        return tuple(self.board), reward, done
    
class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_q(self, state, action):
        return self.q[(state, action)]
        
    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return max(actions, key=lambda a: self.get_q(state, a))
        
    def update(self, state, action, reward, next_state, next_actions, done):
        future = 0 if done else max(self.get_q(next_state, a) for a in next_actions) if next_actions else 0
        self.q[(state, action)] += self.alpha * (reward + self.gamma * future - self.get_q(state, action))
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

if __name__ == "__main__":  # training only runs when you execute train.py directly
    env = TicTacToeEnv()
    agent = QLearningAgent()

    for episode in range(100_000):
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

    print("Training complete!")