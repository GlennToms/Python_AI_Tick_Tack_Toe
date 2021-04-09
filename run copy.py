import tensorflow as tf


(x_train, 


class Game:
    def __init__(self, rows: int, cols: int, auto_reset: bool = False, show_only_end: bool = False, render=False):
        self.rows = rows
        self.cols = cols
        self.state = np.full(shape=(rows, cols), fill_value=None)
        self.player = True
        self.winner = None
        self.auto_reset = auto_reset
        self.game_over = False
        self.show_only_end = show_only_end
        self.render = render

    def reset(self):
        self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
        self.game_over = False
        self.player = True

    def flat_state(self):
        all_values = [y for x in self.state for y in x]
        converted = []
        for index, x in enumerate(all_values):
            if x == "x":
                converted.append(1)
            elif x == "o":
                converted.append(0)
            else:
                converted.append(-1)
        return np.array(converted)

    def step(self, pos: int):
        pos = pos + 1
        row = 0
        col = pos

        # TODO: This is hard codes for 3x3
        if pos >= self.rows:
            row = 1
            col = pos - self.rows
            if pos >= (self.rows * 2):
                row = 2
                col = pos - (self.rows * 2)
        placed_okay = self.place(col, row)
        reward = 1 if placed_okay else -1
        if placed_okay:
            self.place(random.randrange(0, self.rows), random.randrange(0, self.cols))
        observation = self.flat_state()
        done = self.game_over
        if done and self.winner:
            # reward = 10 - len([x for x in observation if x is None])
            reward = 500
        elif done and self.winner is None:
            reward = 0
        elif done and self.winner is False:
            reward = -10 - len([x for x in observation if x is None])
        info = None

        return (observation, reward, done, info)

    def place(self, col: int, row: int) -> bool:
        """Returns True placement was acceptable"""
        self.state = np.array(self.state, dtype=object)
        char = "o"
        if self.player:
            char = "x"

        if self.state[col, row] is None:

            self.state[col, row] = char
            if self.show_only_end is False:
                if self.render:
                    self.print()

            won = self.check_if_game_over()

            if won == -1:
                if self.show_only_end:
                    if self.render:
                        self.print()
                if self.render:
                    print(f"Game is a Draw!")
                self.winner = None
                self.game_over = True
                if self.auto_reset:
                    self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
                    self.game_over = False
                    return False

            elif won is True:
                if self.show_only_end:
                    if self.render:
                        self.print()
                char = "o"
                if self.player:
                    self.winner = self.player
                    char = "x"
                if self.render:
                    print(f"'{char}' has won!")
                self.game_over = True

                if self.auto_reset:
                    self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
                    self.game_over = False
                    return False

                return True

            self.player = not self.player
            return True
        return False

    def print(self):
        if self.render:
            for x in self.state:
                for y in x:
                    if y is None:
                        print(f"|   ", end="")
                    else:
                        print(f"| {y} ", end="")
                print("|")
            print("-------------")

    def check_if_game_over(self) -> bool:
        """returns True if game is over"""
        # Check if row has won.
        for row in self.state:
            if len([x for x in row if x is not None]) == len(row):
                if len(set(row)) == 1:
                    self.game_over = True
                    return True

        # Check if Col has won.
        for row in self.state.transpose():
            if len([x for x in row if x is not None]) == len(row):
                if len(set(row)) == 1:
                    self.game_over = True
                    return True

        # Check if diagonals has won top-left to bottom-right
        diagonal = [self.state[index, index] for index, _ in enumerate(self.state)]
        if len([x for x in diagonal if x is not None]) == self.rows:
            if len(set(diagonal)) == 1:
                self.game_over = True
                return True

        # Check if diagonals has won top-right to bottom-left
        diagonal = [np.fliplr(self.state)[index, index] for index, _ in enumerate(np.fliplr(self.state))]
        if len([x for x in diagonal if x is not None]) == self.rows:
            if len(set(diagonal)) == 1:
                self.game_over = True
                return True

        # Check for free spaces
        if len([y for x in self.state for y in x if y is None]) > 0:
            return False

        # Check if no free spaces left
        if len([y for x in self.state for y in x if y is not None]) > 0:
            return -1

    def __repr__(self):
        return str(self.rows * self.cols)

    def __str__(self):
        return self.__repr__()

    def __bool__(self) -> bool:
        """Returns True while games have moves left or game not won"""
        return not self.game_over


game = Game(3, 3, show_only_end=True)
