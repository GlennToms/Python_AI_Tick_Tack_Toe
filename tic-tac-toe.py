import random
import numpy as np
from collections import Counter
from datetime import datetime

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=200, target_model_update=1e-2
    )
    return dqn


class Game:
    def __init__(self):
        self.rows = 3
        self.cols = 3
        self.actions = self.rows * self.cols
        self.states = self.actions
        self.state = None
        self.done = False
        self.is_ai_turn = True
        self.ai_marker = "x"
        self.player2_marker = "o"
        self.failed_placement = False
        self.points = 0
        self.reset()

    def reset(self):
        self.state = np.full(shape=(self.rows, self.cols), fill_value=None)
        self.done = False
        self.failed_placement = False
        self.points = 0
        return self.flat_state()

    def flat_state(self):
        all_values = [y for x in self.state for y in x]
        flatten = []
        for _, x in enumerate(all_values):
            if x == self.ai_marker:
                flatten.append(2)
            elif x == self.player2_marker:
                flatten.append(1)
            else:
                flatten.append(0)
        return np.array(flatten)

    def step(self, pos: int):
        row = 0
        col = pos

        if pos >= self.rows:
            row = 1
            col = pos - self.rows
            if pos >= (self.rows * 2):
                row = 2
                col = pos - (self.rows * 2)

        if self.place(col, row) and self.done is False:
            reward = 1
            player2_placed = self.place(random.randrange(0, self.rows), random.randrange(0, self.cols))
            while player2_placed is False:
                player2_placed = self.place(random.randrange(0, self.rows), random.randrange(0, self.cols))
                if self.done:
                    break
        else:
            # This is a Hack to make AI lose when it fails it's placements
            self.is_ai_turn = False
            reward = 0
            self.done = True

        observation = self.flat_state()

        if self.done and self.is_ai_turn is True:
            reward = 12 - Counter(observation)[self.ai_marker]
            self.points += reward
        # This checks for loss and Draw conditions
        elif self.done and self.is_ai_turn is False:
            reward = 0
            self.points += reward

        if self.done is False:
            self.points += reward

        return (observation, reward, self.done, {"points": self.points})

    def place(self, col: int, row: int) -> bool:
        """Returns True placement was acceptable"""
        if self.state[col, row] is None:
            if self.is_ai_turn:
                self.state[col, row] = self.ai_marker
            else:
                self.state[col, row] = self.player2_marker
            if self.check_if_game_over():
                return True
            else:
                self.is_ai_turn = not self.is_ai_turn
                return True
        self.check_if_game_over()
        return False

    def render(self):
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
                    self.done = True
                    return True

        # Check if Col has won.
        for row in self.state.transpose():
            if len([x for x in row if x is not None]) == len(row):
                if len(set(row)) == 1:
                    self.done = True
                    return True

        # Check if diagonals has won top-left to bottom-right
        diagonal = [self.state[index, index] for index, _ in enumerate(self.state)]
        if len([x for x in diagonal if x is not None]) == self.rows:
            if len(set(diagonal)) == 1:
                self.done = True
                return True

        # Check if diagonals has won top-right to bottom-left
        diagonal = [np.fliplr(self.state)[index, index] for index, _ in enumerate(np.fliplr(self.state))]
        if len([x for x in diagonal if x is not None]) == self.rows:
            if len(set(diagonal)) == 1:
                self.done = True
                return True

        # Check if no free spaces left
        if len([y for x in self.state for y in x if y is not None]) == self.cols * self.rows:
            self.done = True
            return True

        # Check for free spaces
        if len([y for x in self.state for y in x if y is None]) > 0:
            return False

    def __repr__(self):
        return str(self.rows * self.cols)

    def __str__(self):
        return self.__repr__()

    def __bool__(self) -> bool:
        """Returns True while games have moves left or game not won"""
        return not self.game_over


env = Game()
states = env.states
actions = env.actions

episodes = 1000
scores = []
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0
    info = None

    while not done:
        # env.render()
        action = random.randrange(0, env.actions)
        state, reward, done, info = env.step(action)
        score += reward
    scores.append(score)
    # print("Episode:{} Score:{}".format(episode, score))

print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")
print(f"Score avg: {np.mean(scores)}")
print(Counter(scores))
print(sorted(Counter(scores).items()))
print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")

model = build_model(states, actions)
# model.summary()

print("building agent")
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=["mae"])
dqn.fit(env, nb_steps=50_000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=1000, visualize=False)
print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")
print(f"Score avg: {np.mean(scores.history['episode_reward'])}")
print(Counter(scores.history["episode_reward"]))
print(sorted(Counter(scores.history["episode_reward"]).items()))
print("")
# _ = dqn.test(env, nb_episodes=15, visualize=True)


d = datetime.now()

dqn.save_weights(f"tic-tac-toe-{d.year}-{d.month}-{d.day}--{d.hour}-{d.minute}.h5f", overwrite=True)
