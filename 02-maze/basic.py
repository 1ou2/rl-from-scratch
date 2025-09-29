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
        self.max_tries = 200
        # this will ensure that we get a reward if we find a solution even if it is at the last move
        self.reward = 2*self.max_tries
        self.end_state= (self.maze_size-1,self.maze_size-1)

    def _build_maze(self):
        maze = list()
        for _ in range(self.maze_size):
            maze.append([0]*self.maze_size)
        # TODO add walls, start point and end point
        maze[0][2] = 1
        maze[1][2] = 1

        return maze
    
    def get_state(self):
        return self.state_x, self.state_y

    def step(self,action):
        """Progress one step
        Args: 
        - current state
        - action
        Returns : next_state, reward, is_done
        """
        is_done = False
        reward = -1
        self.nb_tries +=1
        # change current state, depending on action received
        if action == "up" and self.state_y > 0:
            # check it is an empty space and not a wall
            if self.maze[state_x][state_y-1] == 0:
                self.state_y -= 1
        if action == "down" and self.state_y < self.maze_size-1:
            if self.maze[state_x][state_y+1] == 0:
                self.state_y += 1
        if action == "left" and self.state_x > 0:
            if self.maze[state_x-1][state_y] == 0:
                self.state_x -= 1            
        if action == "right" and self.state_x < self.maze_size-1:
            if self.maze[state_x+1][state_y] == 0:
                self.state_x += 1

        if (self.state_x,self.state_y) == self.end_state:
            is_done = True
            reward = 10
        if self.nb_tries >= self.max_tries:
            is_done = True

        return (self.state_x,self.state_y,reward, is_done)


class Agent:
    """
    Handles :
    - policy
    """
    def __init__(self,state_x,state_y):
        self.policy = None
        self.value = dict()
        self.actions = ["up","down","left","right"]
        self.state_x = state_x
        self.state_y = state_y
        self.action = None
        self.gamma = 0.8
        self.epsilon = 0.7

    def choose_action(self):
        # exploit
        if random.random()< self.epsilon:
            print("EXPLOIT")
            # get max values
            vals = []
            max_choices = []
            # get the value of each action in the current state
            for action in self.actions:
                state_action =  f"{self.state_x}|{self.state_y}|{self.action}"
                # no value associated with this action, set a default value of 0
                if self.value.get(state_action) is None:
                    self.value[state_action] = 0
                vals.append(self.value[state_action])

            max_val = max(vals)
            # we can have several choices that have the same max value
            for i, v in enumerate(vals):
                if v == max_val:
                    max_choices.append(self.actions[i])
            # choose an action between the choices having the max value
            self.action = random.choice(max_choices)
        else:
            print("EXPLORE")
            # choose a random action
            self.action = random.choice(self.actions)
        return self.action
    
    def update(self,next_state_x, next_state_y, reward):
        state_action = f"{self.state_x}|{self.state_y}|{self.action}"
        state_reward = self.value.get(state_action,0)
        self.value[state_action] = reward + self.gamma*state_reward
        self.state_x = next_state_x
        self.state_y = next_state_y


if __name__ == "__main__":
    env = Environment()
    is_done = False
    state_x, state_y = env.get_state()
    agent = Agent(state_x,state_y)

    while not is_done:
        action = agent.choose_action()
        state_action = f"{state_x}|{state_y}|{action}"
        print(f"state {state_x,state_y} | action {action} | {agent.value.get(state_action,0.0)}")
        state_x, state_y, reward, is_done = env.step(action)
        agent.update(state_x,state_y,reward)
        print(f"({state_x},{state_y})|{reward}|{is_done}\n")

    print("\n\n")
    for k,v in agent.value.items():
        print(f"{k} --> {v}")