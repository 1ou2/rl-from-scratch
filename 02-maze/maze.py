class Environment:
    def __init__(self, maze):
        self.maze = maze
        self.agent_position = maze.start

    def get_percepts(self):
        return {
            "position": self.agent_position,
            "is_wall": self.maze.is_wall(*self.agent_position),
            "is_end": self.agent_position == self.maze.end,
        }

    def do_action(self, action):
        row, col = self.agent_position
        if action == "up":
            row -= 1
        elif action == "down":
            row += 1
        elif action == "left":
            col -= 1
        elif action == "right":
            col += 1

        if self.maze.is_valid_move(row, col):
            self.agent_position = (row, col)

class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.maze = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = (0, 0)
        self.end = (rows - 1, cols - 1)

    def add_wall(self, row, col):
        self.maze[row][col] = 1

    def set_start(self, row, col):
        self.start = (row, col)
        self.maze[row][col] = 0

    def is_wall(self, row, col):
        return self.maze[row][col] == 1
    
    def is_valid_move(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and not self.is_wall(row, col)
    
    def __repr__(self):
        result = ""
        for row in range(self.rows):
            for col in range(self.cols):
                if (row, col) == self.start:
                    result += "S "
                    continue
                if (row, col) == self.end:
                    result += "E "
                    continue
                result += str(self.maze[row][col]) + " "
            result += "\n"
        return result
    


if __name__ == "__main__":
    maze = Maze(5, 5)
    maze.set_start(4, 0)
    maze.add_wall(3, 2)
    maze.add_wall(4, 2)

    print(maze)