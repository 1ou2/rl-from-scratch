import random

class Gridworld:
    def __init__(self,grid_size=4):
        self.grid_size = grid_size
        self.state_x = 0
        self.state_y = 0
        self.terminal_states = [(0,0),(grid_size-1,grid_size-1)]

    def get_next_state(self,state,action):
        if state in self.terminal_states:
            return state

        x,y = state
        if action == "up" and y > 0:
            y -= 1
        if action == "down" and y < self.grid_size-1:
            y += 1
        if action == "left" and x > 0:
            x -= 1            
        if action == "right" and x < self.grid_size-1:
            x += 1
        return (x,y)

    def get_immediate_reward(self,state,action):
        if state in self.terminal_states:
            return 0
        
        # all actions have a cost of -1
        return -1
        
    def get_accessible_states(self, state):
        """Returns a list of accessible states from the given state."""
        accessible_states = []
        for action in ["up", "down", "left", "right"]:
            next_state = self.get_next_state(state, action)
            if next_state != state:  # Only add if the state changes
                accessible_states.append(next_state)
        return accessible_states
    
    def state_transition_probabilities(self, state):
        """Returns a dictionary of possible next states and their probabilities."""
        # in a terminated state, stay in the same state
        if state in self.terminal_states:
            return {state: 1.0}
        # all neighbour states are equiprobable, with probability 1/4
        # if action would lead to a wall, stay in the same state
        probabilities = {}
        for action in ["up", "down", "left", "right"]:
            next_state = self.get_next_state(state, action)
            if next_state in probabilities:
                probabilities[next_state] += 0.25
            else:
                probabilities[next_state] = 0.25
        return probabilities
            

class RandomAgent:
    """An agent that chooses actions randomly"""
    def __init__(self,grid_size=4):
        self.actions = ["up","down","left","right"]
        self.epsilon = 0.7
        self.grid_size = grid_size
        self.state = (0,0)
        self.value = dict()
        self._initialize_value()

    def _initialize_value(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.value[(i,j)] = 0.0

    def choose_action(self):        
        return random.choice(self.actions)


    def show_values(self):
        for i in range(self.grid_size):
            print("---------------------------")
            out = "| "
            for j in range(self.grid_size):
                v = self.value[(i,j)]
                out += f"{v:+8.03f} | "
            print(out)
        print("---------------------------")



if __name__ == "__main__":
    grid_size = 4
    env = Gridworld(grid_size)
    agent = RandomAgent(grid_size)
    gamma = 1.0

    agent.show_values()

    for it in range(100):
        print(f"ITERATION {it+1}")
        
        new_value = agent.value.copy()
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i,j)
                new_value[state] = 0
                for action in agent.actions:
                    reward = env.get_immediate_reward(state,action)
                    next_state = env.get_next_state(state,action)
                    # For random agent, probability associated to an action is: 1/len(agent.actions)
                    new_value[state] += (1/len(agent.actions))*(reward + gamma*agent.value[next_state])
        agent.value = new_value
        agent.show_values()