import random

class Environment:
    """
    Handles :
    - current state
    - is episode done
    - maze structure
    - reward system

    """
    def __init__(self,size=5):
        self.maze_size = size
        self.state_x = 0
        self.state_y = 0
        self.current_state = (0,0)
        self.maze = self._build_maze()
        self.nb_tries = 0
        self.max_tries = 12
        self.end_state= (self.maze_size-1,self.maze_size-1)

    def _build_maze(self):
        maze = list()
        for _ in range(self.maze_size):
            maze.append([0]*self.maze_size)
        # TODO add walls, start point and end point

        return maze
    
    def step(self,action):
        """Progress one step
        Args: 
        - current state
        - action
        Returns : next_state, reward, is_done
        """
        is_done = False
        reward = -1
        # change current state, depending on action received
        if action == "up" and self.state_y > 0:
            self.state_y -=1

        # TODO : for all actions
        if (self.state_x,self.state_y) == self.end_state:
            is_done = True
            reward = 10
        

        return (self.state_x,self.state_y,reward, is_done)


class Agent:
    """
    Handles :
    - policy
    """
    def __init__(self):
        self.policy = None
        self.actions = ["up","down","left","right"]

    def action(self,state_x,state_y):
        pass


if __name__ == "__main__":
    env = Environment()
    print(env.maze)
        