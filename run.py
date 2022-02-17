import random
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter


LR = 1e-3
goal_steps = 25
score_requirement = 4
initial_games = 100_000
WINS = 0
NOT_WINS = 0


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
        for _, x in enumerate(all_values):
            if x == "x":
                converted.append(2)
            elif x == "o":
                converted.append(1)
            else:
                converted.append(0)
        return np.array(converted)

    def step1(self, pos: int):
        return self.step(pos - 1)

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
        else:
            return (self.flat_state(), 0, True, None)
        observation = self.flat_state()
        done = self.game_over
        if done and self.winner is True:
            reward = Counter(observation)[2] + 2
            global WINS
            WINS = WINS + 1
        # This checks for loss and Draw conditions
        elif done and self.winner is not True:
            reward = 0
            global NOT_WINS
            NOT_WINS = NOT_WINS + 1
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
                self.winner = -1
                self.game_over = True
                if self.auto_reset:
                    self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
                    self.game_over = False
                    return False

            elif won is True:
                if self.show_only_end:
                    if self.render:
                        self.print()
                char = "Random"
                self.winner = self.player
                if self.player:
                    char = "AI"
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


game = Game(3, 3, show_only_end=False, render=False)


def initial_population():
    # [OBS, MOVES]
    training_data = []
    # all scores:
    scores = []
    # just the scores that met our threshold:
    accepted_scores = []
    # iterate through however many games we want:
    for i in range(initial_games):
        print(f"Game: {initial_games - i}")
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0, (game.rows * game.cols) - 1)
            # do it!
            observation, reward, done, info = game.step(action)

            # notice that the observation is returned FROM the action
            # so we'll store the previous observation here, pairing
            # the prev observation to the action we'll take.
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done:
                break

        # IF our score is higher than our threshold, we'd like to save
        # every move we made
        # NOTE the reinforcement methodology here.
        # all we're doing is reinforcing the score, we're not trying
        # to influence the machine in any way as to HOW that score is
        # reached.
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 0:
                    output = [1, 0, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 1:
                    output = [0, 1, 0, 0, 0, 0, 0, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1, 0, 0, 0, 0, 0, 0]
                elif data[1] == 3:
                    output = [0, 0, 0, 1, 0, 0, 0, 0, 0]
                elif data[1] == 4:
                    output = [0, 0, 0, 0, 1, 0, 0, 0, 0]
                elif data[1] == 5:
                    output = [0, 0, 0, 0, 0, 1, 0, 0, 0]
                elif data[1] == 6:
                    output = [0, 0, 0, 0, 0, 0, 1, 0, 0]
                elif data[1] == 7:
                    output = [0, 0, 0, 0, 0, 0, 0, 1, 0]
                elif data[1] == 8:
                    output = [0, 0, 0, 0, 0, 0, 0, 0, 1]

                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        game.reset()
        # save overall scores
        scores.append(score)

    # just in case you wanted to reference later
    training_data_save = np.array(training_data)
    np.save("saved.npy", training_data_save)

    # some stats here, to further illustrate the neural network magic!
    print("Average accepted score:", mean(accepted_scores))
    print("Median score for accepted scores:", median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name="input")

    network = fully_connected(network, 128, activation="relu")
    # network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation="relu")
    network = dropout(network, 0.8)

    # network = fully_connected(network, 512, activation="relu")
    # network = dropout(network, 0.8)

    # network = fully_connected(network, 256, activation="relu")
    # network = dropout(network, 0.8)

    # network = fully_connected(network, 128, activation="relu")
    # network = dropout(network, 0.8)

    network = fully_connected(network, 9, activation="softmax")
    network = regression(network, optimizer="adam", learning_rate=LR, loss="categorical_crossentropy", name="targets")
    model = tflearn.DNN(network, tensorboard_dir="log")

    return model


def train_model(training_data, model=False):

    X = np.array([i[0] for i in training_data], dtype=np.float).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({"input": X}, {"targets": y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id="tic-tac-toe")
    return model


training_data = initial_population()
model = train_model(training_data)
scores = []
choices = []
for _ in range(10_000):
    # print("Starting New Game")
    print(f"WINS: {WINS}, NOT_WINS: {NOT_WINS}, GAMES:{WINS + NOT_WINS}")
    score = 0
    game_memory = []
    prev_obs = []
    game.reset()
    for _ in range(goal_steps):
        if len(prev_obs) == 0:
            action = random.randrange(0, 8)
        else:
            action = int(np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0]))
            choices.append(action)
        # print(action)

        new_observation, reward, done, info = game.step1(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break

    scores.append(score)

print("Average Score:", sum(scores) / len(scores))
print(f"Scores: {Counter(scores)}")
print(f"Actions: {Counter(choices)}")
