# 🎮 Maze Solver with Q-Learning

An interactive web-based maze solver that uses Deep Q-Learning to find optimal paths through randomly generated mazes. Watch an AI agent learn to navigate through obstacles and find the shortest path to the goal!

## ✨ Features

- 🎲 Random maze generation with guaranteed solvable paths
- 🤖 Deep Q-Learning agent that learns optimal navigation strategies
- 🎯 Interactive web interface to visualize the maze and solution
- 🔄 Adjustable maze sizes
- 📊 Real-time training progress visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ 
- Node.js (for running the frontend)
- Web browser

### Installation

1. Clone the repository:
```bash
git clone https://github.com/QBitScripter/maze_solver
cd maze-solver-qlearning
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate

pip install -r requirements.txt
```

3. Run the backend server:
```bash
python app.py
```

4. Open the frontend:
- Navigate to the frontend directory
- Open `index.html` in your web browser

## 🎮 How to Use

1. Click "Initialize Maze" to generate a new random maze
2. Click "Train Agent" to start the Q-learning training process
3. Once training is complete, click "Get Solution" to see the agent navigate to the goal
4. Use "Refresh Maze" to generate a different maze layout

## 🧠 Technical Details

### Backend
- Flask server for handling maze operations and training
- Custom Q-learning implementation using PyTorch
- Maze environment with configurable size and obstacle density

### Frontend
- Pure HTML/CSS/JavaScript implementation
- Real-time visualization of maze and agent movement
- Responsive grid-based maze display

## 🛠️ Project Structure

```
maze-solver-qlearning/
├── backend/
│   ├── app.py                 # Flask server
│   ├── requirements.txt       # Python dependencies
│   └── qlearning/
│       ├── agent.py          # Q-learning agent implementation
│       └── environment.py    # Maze environment
├── frontend/
│   ├── src/
│   │   ├── index.html       # Main webpage
│   │   ├── style.css        # Styling
│   │   └── main.js         # Frontend logic
│   └── public/
└── README.md
```




## 🤝 Contributing

Feel free to open issues and pull requests!

## ✨ Acknowledgments

- Built with PyTorch
- Inspired by Q-learning and reinforcement learning principles
- Special thanks to the Flask and CORS teams for making the web integration smooth
