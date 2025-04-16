from game import MazeGame
import numpy as np
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration

def main():

    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0]
    ])
    """
    rewards = np.array([
        [1,    1,   1,   1,   1,   1,   1,   1,      1],
        [1,  -10,  -9,  -8,   1,  -8,   1,  -4,     -3],
        [1,  -11,   1,  -9,  -8,  -7,   1,   1,     -2],
        [1,  -12,   1,  -8,   1,  -6,  -5,  -2,     -1],
        [1,    1,   1,   1,   1,   1,   1,   1,    0]
    ])
    """
    goal = (maze.shape[0]-1, maze.shape[1]-1)
    rewards = -1*np.ones_like(maze)
    rewards[goal] = 100
    game = MazeGame(
        maze, 
        rewards,
        initial_state=(maze.shape[0]-1,1),
        goal_state=goal
    )
    
    print("Maze")
    print(maze)
    
    # Value Iteration
    max_iteration = 7_000
    VI = ValueIteration(game, 0.9, 1e-8, max_iteration=max_iteration)
    
    # value iteration process
    VI.value_iteration()
    
    print("Value Function:")
    print(VI.V)
    
    # policy extraction from values
    VI.policy_extraction()
    
    print("Policy:")
    print(VI.P)

    # run calculated plicy
    frames = VI.run_policy(max_try=1000)
    
    VI.save_video(frames, "value_iteration_maze_solution.mp4")
    
    # Policy Iteration
    PI = PolicyIteration(
        game,
        0.9,
        1e-6,
        7500
    )
    
    PI.policy_iteration()
    
    print("Value Function:")
    print(PI.V)
    
    print("Policy:")
    print(PI.P)
    
    frames = PI.run_policy(1000)
    
    PI.save_video(frames, "policy_iteration_maze_solution.mp4")
    


if __name__ == "__main__":
    main()
