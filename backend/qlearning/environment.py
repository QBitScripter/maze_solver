import numpy as np
from queue import Queue


class MazeEnvironment:
    def __init__(self, size=7):
        """
        Initializes the maze environment.

        :param size: The size of the maze (size x size grid)
        """
        self.size = size
        self.start = (0, 0)  # Starting position
        self.goal = (size - 1, size - 1)  # Goal position
        self.reset()

    
    def is_valid_path(self):
        """
        Checks if there is a valid path from start to goal using BFS.
        Returns True if a path exists, False otherwise.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        visited = set()
        queue = Queue()
        queue.put(self.start)
        visited.add(self.start)

        while not queue.empty():
            x, y = queue.get()

            # If we reach the goal, a valid path exists
            if (x, y) == self.goal:
                return True

            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # Check bounds and walls
                if 0 <= nx < self.size and 0 <= ny < self.size and self.maze[nx, ny] != 1 and (nx, ny) not in visited:
                    queue.put((nx, ny))
                    visited.add((nx, ny))

        # If we exhaust all possibilities, no path exists
        return False
    
    # def reset(self):
    #     """
    #     Resets the maze environment.
    #     """
    #     self.start = (np.random.randint(0, self.size), np.random.randint(0, self.size))
    #     self.goal = (np.random.randint(0, self.size), np.random.randint(0, self.size))
    
    #     while self.start == self.goal:  # Ensure start and goal are not the same
    #         self.goal = (np.random.randint(0, self.size), np.random.randint(0, self.size))
        
    #     self.maze = np.zeros((self.size, self.size), dtype=int)
    #     self.maze[self.start] = 2  # Mark the start position
    #     self.maze[self.goal] = 3  # Mark the goal position

    #     # Add walls (obstacles) randomly
    #     num_walls = int(self.size * self.size * 0.7)  # 30% of the grid as walls
    #     for _ in range(num_walls):
    #         x, y = np.random.randint(0, self.size, 2)
    #         if (x, y) != self.start and (x, y) != self.goal:
    #             self.maze[x, y] = 1  # Mark as a wall
    
    #     self.agent_position = self.start  # Reset agent's position to the start
    #     return self.flatten_state(self.agent_position)
    
    
    def reset(self):
        """
        Resets the maze environment with random start and goal positions.
        Ensures that a valid path exists between start and goal.
        """
        while True:  # Keep regenerating the maze until a valid path exists
            # Randomize start and goal positions
            self.start = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            self.goal = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            
            # Ensure start and goal positions are not the same
            while self.start == self.goal:
                self.goal = (np.random.randint(0, self.size), np.random.randint(0, self.size))

            # Generate maze and add walls
            self.maze = np.zeros((self.size, self.size), dtype=int)
            self.maze[self.start] = 2  # Mark the start position
            self.maze[self.goal] = 3  # Mark the goal position

            num_walls = int(self.size * self.size * 0.4)  # 40% of the grid as walls
            for _ in range(num_walls):
                x, y = np.random.randint(0, self.size, 2)
                if (x, y) != self.start and (x, y) != self.goal:
                    self.maze[x, y] = 1  # Mark as a wall

            # Check if a valid path exists
            if self.is_valid_path():
                break  # Exit the loop only if the path is valid

        self.agent_position = self.start  # Reset agent's position to the start
        return self.flatten_state(self.agent_position)


    def flatten_state(self, state):
        """
        Converts the maze state into a flattened one-hot encoded vector.

        :param state: Tuple (x, y) representing the current position
        :return: Flattened one-hot encoded vector of the maze
        """
        flat_state = np.zeros(self.size * self.size)
        flat_state[state[0] * self.size + state[1]] = 1
        return flat_state
    def step(self, action, log_file="training_log.txt"):
        with open(log_file, "w") as log:
            """
            Move the agent based on the action and return reward and done flag.
            """
            x, y = self.agent_position

            # Determine new position based on the action
            if action == 0:  # Up
                new_position = (x - 1, y)
            elif action == 1:  # Right
                new_position = (x, y + 1)
            elif action == 2:  # Down
                new_position = (x + 1, y)
            elif action == 3:  # Left
                new_position = (x, y - 1)
            else:
                log.write(f"New position: {new_position}, Reward: {reward}")
                return self.flatten_state(self.agent_position), -1, False  # Invalid action

            # Check if the new position is out of bounds or a wall
            if (
                new_position[0] < 0
                or new_position[1] < 0
                or new_position[0] >= self.size
                or new_position[1] >= self.size
                or self.maze[new_position] == 1  # Wall
            ):
                return self.flatten_state(self.agent_position), -1, False  # Penalize invalid move

            # Reward for a valid move (slight negative reward to encourage efficiency)
            reward = -0.05

            # Optional: Reward progress toward goal (e.g., Manhattan distance)
            current_distance = abs(self.agent_position[0] - self.goal[0]) + abs(self.agent_position[1] - self.goal[1])
            new_distance = abs(new_position[0] - self.goal[0]) + abs(new_position[1] - self.goal[1])

            # If the agent moved closer to the goal, give a small reward
            if new_distance < current_distance:
                reward = 0.05  # Slight positive reward for moving closer to the goal
            
            # Check if the agent reached the goal
            if new_position == self.goal:
                self.agent_position = new_position  # Update position
                return self.flatten_state(new_position), 20, True  # Reward for reaching the goal

            # Update agent's position for a valid move
            self.agent_position = new_position

            return self.flatten_state(new_position), reward, False  # Return state, reward, done


    def render(self):
        """
        Renders the current state of the maze.
        """
        display_maze = self.maze.copy()
        x, y = self.agent_position
        display_maze[x, y] = 9  # Mark the agent's current position
        print(f"Displaying Maze:\n{display_maze}")
