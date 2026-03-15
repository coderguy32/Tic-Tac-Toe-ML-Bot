from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pickle, random, os, threading
from collections import defaultdict
from ttt_train import TicTacToeEnv, QLearningAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Looking for index.html in:", BASE_DIR)
print("Files found:", os.listdir(BASE_DIR))

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)
 
env = TicTacToeEnv()
agent = QLearningAgent()
 
if os.path.exists("qtable.pkl"):
    with open("qtable.pkl", "rb") as f:
        agent.q = defaultdict(float, pickle.load(f))
    print("Loaded saved Q-table.")
 
# Training runs in a background thread so the server stays responsive
train_state = {
    "running": False,
    "episodes_done": 0,
    "total_episodes": 0,
}
 
def save():
    with open("qtable.pkl", "wb") as f:
        pickle.dump(dict(agent.q), f)
 
def run_training(episodes):
    agent.epsilon = 1.0
    train_state["running"] = True
    train_state["episodes_done"] = 0
    train_state["total_episodes"] = episodes
 
    for episode in range(episodes):
        if not train_state["running"]:
            break
 
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
 
        train_state["episodes_done"] = episode + 1
 
    train_state["running"] = False
    save()
    print(f"Training complete. {train_state['episodes_done']} episodes run.")
 
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')
 
@app.route('/train', methods=['POST'])
def train():
    if train_state["running"]:
        return jsonify({"error": "Already training"}), 400
    data = request.json
    episodes = int(data.get("episodes", 100000))
    alpha = float(data.get("alpha", 0.5))
    gamma = float(data.get("gamma", 0.9))
    decay = float(data.get("epsilon_decay", 0.9995))
    agent.alpha = alpha
    agent.gamma = gamma
    agent.epsilonDecay = decay
    t = threading.Thread(target=run_training, args=(episodes,), daemon=True)
    t.start()
    return jsonify({"status": "started", "episodes": episodes})
 
@app.route('/stop', methods=['POST'])
def stop():
    train_state["running"] = False
    return jsonify({"status": "stopped"})
 
@app.route('/progress', methods=['GET'])
def progress():
    return jsonify({
        "running": train_state["running"],
        "episodes_done": train_state["episodes_done"],
        "total_episodes": train_state["total_episodes"],
        "epsilon": round(agent.epsilon, 4),
        "q_entries": len(agent.q),
    })
 
@app.route('/move', methods=['POST'])
def move():
    data = request.json
    board = data.get("board")  # list of 9 ints: 0, 1, -1
    env.board = board
    actions = env.available_actions()
    if not actions:
        return jsonify({"error": "No moves available"}), 400
    agent.epsilon = 0  # always play best move
    action = agent.choose_action(tuple(board), actions)
    return jsonify({"move": action})
 
@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        "q_entries": len(agent.q),
        "epsilon": round(agent.epsilon, 4),
        "trained": len(agent.q) > 0,
    })
 
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
