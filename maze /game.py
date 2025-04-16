import numpy as np
from gymnasium import spaces
import gymnasium as gym
from PIL import Image, ImageDraw


class MazeGame(gym.Env):
   
    
    def __init__(
        self, 
        maze: np.ndarray, 
        rewards: np.ndarray,
        initial_state: tuple,
        goal_state: tuple
    ) -> None:
        self.maze=maze
        self.n_rows, self.n_cols = maze.shape
        
        self.start_state = initial_state # (2,1)
        self.goal_state = goal_state # (self.n_rows-1, self.n_cols-1)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.n_rows), 
            spaces.Discrete(self.n_cols)
        ))
        self.rewards=rewards
        self.action_space = spaces.Discrete(4)
        self.state=self.start_state
        
    def step(self, action):
        # Define actions: 0=up, 1=down, 2=left, 3=right
        row, col = self.state
        if action == 0 and row > 0 and self.maze[row-1, col] == 0:  # Up
            row -= 1
        elif action == 1 and row < self.n_rows-1 and self.maze[row+1, col] == 0:  # Down
            row += 1
        elif action == 2 and col > 0 and self.maze[row, col-1] == 0:  # Left
            col -= 1
        elif action == 3 and col < self.n_cols-1 and self.maze[row, col+1] == 0:  # Right
            col += 1
        
        next_state = (row, col)
        
        reward = self.rewards[next_state]  # Default step cost
        done = next_state == self.goal_state
        self.state=next_state
        return next_state, reward, done, {}
    
    
    def reset(self):
        self.state = self.start_state
        return self.state 
      
    def _render_rgb_array(self, P):
        # Create an RGB image of the maze
        cell_size = 50  # Size of each cell in pixels
        img_size = (self.n_cols * cell_size, self.n_rows * cell_size)
        img = Image.new("RGB", img_size, "white")
        draw = ImageDraw.Draw(img)
        
        # Draw walls
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                
                top_left = (c * cell_size, r * cell_size)
                bottom_right = ((c + 1) * cell_size, (r + 1) * cell_size)
                if self.maze[r, c] == 1:  # Wall
                    draw.rectangle([top_left, bottom_right], fill="black")
                elif (r,c) == self.goal_state: # Goal
                    draw.rectangle([top_left, bottom_right], fill="red")
                else:
                    # Draw Arrows
                    action = P[r, c]
                    center = (c * cell_size + cell_size // 2, r * cell_size + cell_size // 2)
                    arrow_length = cell_size // 2

                    if action == 0:  # Up
                        draw.line([center, (center[0], center[1] - arrow_length)], fill="green", width=3)
                        draw.polygon([
                            (center[0] - 5, center[1] - arrow_length + 10),
                            (center[0] + 5, center[1] - arrow_length + 10),
                            (center[0], center[1] - arrow_length)
                        ], fill="green")
                    elif action == 3:  # Right
                        draw.line([center, (center[0] + arrow_length, center[1])], fill="green", width=3)
                        draw.polygon([
                            (center[0] + arrow_length - 10, center[1] - 5),
                            (center[0] + arrow_length - 10, center[1] + 5),
                            (center[0] + arrow_length, center[1])
                        ], fill="green")
                    elif action == 1:  # Down
                        draw.line([center, (center[0], center[1] + arrow_length)], fill="green", width=3)
                        draw.polygon([
                            (center[0] - 5, center[1] + arrow_length - 10),
                            (center[0] + 5, center[1] + arrow_length - 10),
                            (center[0], center[1] + arrow_length)
                        ], fill="green")
                    elif action == 2:  # Left
                        draw.line([center, (center[0] - arrow_length, center[1])], fill="green", width=3)
                        draw.polygon([
                            (center[0] - arrow_length + 10, center[1] - 5),
                            (center[0] - arrow_length + 10, center[1] + 5),
                            (center[0] - arrow_length, center[1])
                        ], fill="green")
        
        # Draw agent
        agent_top_left = (self.state[1] * cell_size, self.state[0] * cell_size)
        agent_bottom_right = ((self.state[1] + 1) * cell_size, 
                              (self.state[0] + 1) * cell_size)
        draw.rectangle([agent_top_left, agent_bottom_right], fill="blue")
        
        
        # Convert to NumPy array
        return np.array(img)
    
    def _render_human(self):
        # Simple text-based rendering for debugging
        render_grid = self.maze.copy().astype(str)
        r, c = self.agent_position
        render_grid[r, c] = "A"
        return "\n".join(" ".join(row) for row in render_grid)
    
    