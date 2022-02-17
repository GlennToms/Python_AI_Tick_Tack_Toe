import random
import numpy as np
from collections import Counter

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
        model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2
    )
    return dqn


class Game:
    def __init__(self, cols: int, num_of_x: int = 1):
        self.cols = cols
        self.actions = self.cols
        self.state = None
        self.done = False
        self.num_of_x = num_of_x
        self.points = 0

        self.marked = 1
        self.filler = 0
        self.do_not_hit = 2

        self.reset()

    def reset(self):
        self.done = False
        self.points = 0
        self.state = np.full(shape=(self.cols), fill_value=self.filler)
        self.state[random.randrange(0, self.cols)] = self.do_not_hit
        return self.state

    def step(self, pos: int):
        placed_okay = self.place(pos)
        observation = self.state
        if placed_okay:
            reward = 1
        else:
            reward = 0
            self.done = True
        self.points += reward
        return (observation, reward, self.done, {})

    def place(self, col: int) -> bool:
        """Returns True placement was acceptable"""
        if self.state[col] == self.filler:
            self.state[col] = self.marked
            return True
        return False

    def render(self):
        line = ""
        for index in range(0, len(self.state)):
            if self.state[index] == self.filler:
                line += str(f"|   ")
            elif self.state[index] == self.marked:
                line += str(f"| o ")
            else:
                line += str("| x ")
        line += str(f"| {self.points} pts")
        print(line)
        print("-" * len(line))

    def check_if_game_over(self) -> bool:
        """returns True if game is over"""
        # Check for free spaces
        if len([y for x in self.state for y in x if y is None]) > 0:
            return False

        # Check if no free spaces left
        if len([y for x in self.state for y in x if y is not None]) > 0:
            self.done = True
            return True

    def __repr__(self):
        return str(self.rows * self.cols)

    def __str__(self):
        return self.__repr__()

    def __bool__(self) -> bool:
        """Returns True while games have moves left or game not won"""
        return not self.done


env = Game(cols=21)
states = env.actions
actions = env.actions

episodes = 1000
scores = []
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = random.randrange(0, env.actions - 1)
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

from datetime import datetime

d = datetime.now()

dqn.save_weights(f"game.h5f", overwrite=True)
dqn.save_weights(f"dont_click_game_{d.year}-{d.month}-{d.day}.h5f", overwrite=True)
