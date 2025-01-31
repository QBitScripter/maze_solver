# app.py
from flask import Flask, jsonify
from qlearning.environment import MazeEnvironment
from qlearning.agent import DQNAgent
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

env = None
agent = None

@app.route('/initialize', methods=['POST'])
@app.route('/initialize/<int:size>', methods=['POST'])
def initialize(size=None):
    global env, agent
    size = size or 7
    env = MazeEnvironment(size)
    agent = DQNAgent(env)
    env.reset()
    return jsonify({"maze": env.maze.tolist()})



@app.route("/current_maze", methods=["GET"])
def get_current_maze():
    if env is None or env.maze is None:
        return jsonify({"error": "Maze not initialized."}), 400
    return jsonify({"maze": env.maze})


@app.route('/train', methods=['POST'])
def train():
    """
    Trains the Q-learning agent if the maze is initialized.
    """
    global agent, env
    if env is None or agent is None:
        return jsonify({"error": "Maze not initialized. Please initialize the maze first."}), 400

    try:
        # Train the agent
        agent.train(episodes=1000)
        return jsonify({"message": "Training complete"})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/solution', methods=['GET'])
def solution():
   def solution():
    if agent is None or env is None:
        return jsonify({"error": "Agent or environment is not initialized."}), 400

    try:
        # Get the solution path from the trained DQN agent
        path = agent.get_solution_path()
        return jsonify({"path": path})
    except Exception as e:
        print(f"Error fetching solution: {e}")
        return jsonify({"error": "Failed to fetch solution path."}), 500
if __name__ == "__main__":
    app.run(debug=True)
