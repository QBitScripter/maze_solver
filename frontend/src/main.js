document.addEventListener("DOMContentLoaded", () => {
    const mazeContainer = document.getElementById("maze-container");

    // Existing Buttons
    document.getElementById("initialize-btn").addEventListener("click", initializeMaze);
    document.getElementById("train-btn").addEventListener("click", trainAgent);
    document.getElementById("solve-btn").addEventListener("click", getSolution);
    document.getElementById("refresh-btn").addEventListener("click", refreshMaze);

    const trainingProgress = document.getElementById("training-progress");  // Add a div for progress display

    function initializeMaze() {
        fetch('http://127.0.0.1:5000/initialize', { method: "POST" })
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to initialize the maze");
            }
            return response.json();
        })
        .then(data => renderMaze(data.maze))
        .catch(error => console.error("Error initializing maze:", error));
    }

    function refreshMaze() {
        // Generate a random size for the maze
        const randomSize = Math.floor(Math.random() * 10) + 5;  // Random size between 5 and 14

        fetch(`http://127.0.0.1:5000/initialize/${randomSize}`, { method: "POST" })
            .then(response => response.json())
            .then(data => renderMaze(data.maze));
    }

    function trainAgent() {
        // Show training progress (initial status)
        trainingProgress.style.display = "block";
        trainingProgress.innerText = "Training started...";

        fetch('http://127.0.0.1:5000/train', { method: 'POST' })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to train the agent");
                }
                return response.json();
            })
            .then(data => {
                // Handle training completion
                trainingProgress.innerText = "Training complete!";
                alert(data.message);
            })
            .catch(error => {
                // Error during training
                trainingProgress.innerText = "Training failed.";
                console.error("Error during training:", error);
            });
    }

    function getSolution() {
        fetch('http://127.0.0.1:5000/solution', { method: "GET" })  // Correct backend URL
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to fetch solution");
                }
                return response.json();
            })
            .then(data => animatePath(data.path))
            .catch(error => console.error("Error fetching solution:", error));
    }

    function renderMaze(maze) {
        mazeContainer.innerHTML = "";
        mazeContainer.style.gridTemplateColumns = `repeat(${maze.length}, 40px)`;

        maze.forEach((row, x) => {
            row.forEach((cell, y) => {
                const cellElement = document.createElement("div");
                cellElement.classList.add("cell");

                if (cell === 1) cellElement.classList.add("wall");
                else if (cell === 2) cellElement.classList.add("start");
                else if (cell === 3) cellElement.classList.add("goal");
                else cellElement.classList.add("path");

                mazeContainer.appendChild(cellElement);
            });
        });
    }

    function animatePath(path) {
        if (!path || path.length === 0) {
            alert("No solution path found.");
            return;
        }
        
        let i = 0;

        function moveAgent() {
            if (i >= path.length) return;

            const [x, y] = path[i];
            const cells = mazeContainer.children;
            Array.from(cells).forEach(cell => cell.classList.remove("agent"));

            const index = x * Math.sqrt(cells.length) + y;
            cells[index].classList.add("agent");

            i++;
            setTimeout(moveAgent, 500);
        }

        moveAgent();
    }
});
