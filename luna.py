# Landing pad is always at coordinates (0,0). Coordinates are the first
# two numbers in state vector. Reward for moving from the top of the screen
# to landing pad and zero speed is about 100..140 points. If lander moves
# away from landing pad it loses reward back. Episode finishes if the lander
# crashes or comes to rest, receiving additional -100 or +100 points.
# Each leg ground contact is +10. Firing main engine is -0.3 points each frame.
# Solved is 200 points. Landing outside landing pad is possible. Fuel is
# infinite, so an agent can learn to fly and then land on its first attempt.
# Four discrete actions available: do nothing, fire left orientation engine,
# fire main engine, fire right orientation engine.


from typing import Counter
import gym
import random
from keras import Sequential
from collections import deque
import keras
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
from tensorflow.python.keras.layers.core import Dropout


class Game:
    def __init__(
        self,
        rows: int,
        cols: int,
        auto_reset: bool = False,
        show_only_end: bool = False,
        render: bool = False,
        human=False,
    ):
        self.rows = rows
        self.cols = cols
        self.state = np.full(shape=(rows, cols), fill_value=None)
        self.player = True
        self.winner = None
        self.auto_reset = auto_reset
        self.game_over = False
        self.show_only_end = show_only_end
        self.render = render
        self.use_this_winner = 0
        self.human = human

    def reset(self):
        self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
        self.game_over = False
        self.player = True
        return self.flat_state()

    def flat_state(self):
        all_values = [y for x in self.state for y in x]
        converted = []
        for x in all_values:
            if x == "x":
                converted.append(2)
            elif x == "o":
                converted.append(1)
            else:
                converted.append(0)
        return np.array(converted)

    def step1(self, pos: int):
        # import time
        # time.sleep(0.2)

        a = self.step(pos - 1)
        return a

    def step(self, pos: int):
        pos += 1
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
        reward = 1 if placed_okay else -5

        done = self.game_over

        if placed_okay and done is False:
            if self.human:
                i = input()
                a = int(i[0]) - 1
                b = int(i[1]) - 1
                self.place(row=a, col=b)
            else:
                self.place(random.randrange(0, self.rows), random.randrange(0, self.cols))
            done = self.game_over
            observation = self.flat_state()
            info = None
            if done:
                if self.use_this_winner == "AI":
                    reward = 10
                elif self.use_this_winner == "None":
                    reward = 0
                elif self.use_this_winner == "RANDOM":
                    reward = -10

                info = self.use_this_winner
                if self.render:
                    print("------- GAME END -------")
                    print()
            return (observation, reward, done, info)

        observation = self.flat_state()
        info = None
        if done:
            if self.use_this_winner == "AI":
                reward = 10
            elif self.use_this_winner == "None":
                reward = 0
            elif self.use_this_winner == "RANDOM":
                reward = -10

            info = self.use_this_winner
            if self.render:
                print("------- GAME END -------")
                print()
        return (observation, reward, done, info)

    def place(self, col: int, row: int) -> bool:
        """Returns True placement was acceptable"""
        self.state = np.array(self.state, dtype=object)
        char = "o"
        if self.player:
            char = "x"

        if self.state[col, row] is None:

            self.state[col, row] = char
            if self.show_only_end is False and self.render:
                self.print()

            won = self.check_if_game_over()

            if won == -1:
                if self.show_only_end and self.render:
                    self.print()
                if self.render:
                    print(f"Game is a Draw!")
                self.use_this_winner = "None"
                self.winner = None
                self.game_over = True
                if self.auto_reset:
                    self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
                    self.game_over = False
                    return False

            elif won is True:
                if self.show_only_end and self.render:
                    self.print()
                char = "RANDOM"
                if self.player:
                    self.winner = self.player
                    char = "AI"
                self.use_this_winner = char
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
            if len([x for x in row if x is not None]) == len(row) and len(set(row)) == 1:
                self.game_over = True
                return True

        # Check if Col has won.
        for row in self.state.transpose():
            if len([x for x in row if x is not None]) == len(row) and len(set(row)) == 1:
                self.game_over = True
                return True

        # Check if diagonals has won top-left to bottom-right
        diagonal = [self.state[index, index] for index, _ in enumerate(self.state)]
        if len([x for x in diagonal if x is not None]) == self.rows and len(set(diagonal)) == 1:
            self.game_over = True
            return True

        # Check if diagonals has won top-right to bottom-left
        diagonal = [np.fliplr(self.state)[index, index] for index, _ in enumerate(np.fliplr(self.state))]
        if len([x for x in diagonal if x is not None]) == self.rows and len(set(diagonal)) == 1:
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


env = Game(3, 3, show_only_end=False, render=False)
env.action_space = 9
# env = gym.make("LunarLander-v2")
# env.seed(0)
np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = 0.99
        self.batch_size = 64
        self.epsilon_min = 0.01
        self.lr = 0.001
        self.epsilon_decay = 0.996
        self.memory = deque(maxlen=1000000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(64, input_dim=self.state_space, activation=relu))
        model.add(Dense(128, activation=relu))
        model.add(Dense(64, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        # import time

        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # model.save(f"tic-tac-toe.keras {timestr}")
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        # import time

        # timestr = time.strftime("%Y%m%d-%H%M%S")

        # self.model.save(timestr)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):
    loss = []
    winners = []
    agent = DQN(env.action_space, 9)
    max_steps = 50
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, len(state)))
        score = 0
        steps = 0
        last_action = None
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step1(action)
            score += reward
            next_state = np.reshape(next_state, (1, 9))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if last_action == action:
                print(last_action, action)
            last_action = action
            agent.replay()
            steps += 1
            if done:
                print(f"episode: {e}/{episode}, Steps: {steps}, Score: {score} ")
                winners.append(info)
                break
            if step == (max_steps - 1):
                print("Maxed Out")
        loss.append(score)
    return loss, winners


def trainer():
    import time

    start_time = time.time()

    episodes = 2000
    loss, winners = train_dqn(episodes)

    print(f"Winners: {Counter(winners)}")
    plt.plot([i + 1 for i in range(0, episodes, 2)], loss[::2])
    # plt.show()
    plt.savefig("tic-tac-toe.png")
    print("--- %s seconds ---" % (time.time() - start_time))


def play():
    model = keras.models.load_model("tic-2")
    env = Game(rows=3, cols=3, render=True, human=True)
    state = env.reset()
    while env:
        action = np.argmax(model.predict(np.reshape(state, (1, len(state)))))
        state, reward, done, info = env.step1(action)
        # print(action)


if __name__ == "__main__":
    # trainer()
    play()
