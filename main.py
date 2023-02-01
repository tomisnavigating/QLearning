import copy
import numpy as np


class MazeEnvironment:

    def __init__(self, w, h, start, goal):
        self.width = w
        self.height = h
        self.start = start
        self.goal = goal
        self.state = start
        self.done = False
        self.penalty = -1
        self.reward = 10

        self.roadblocks = [(0, 5),
                           (0, 7),
                           (0, 9),
                           (1, 1),
                           (1, 2),
                           (1, 3),
                           (1, 5),
                           (2, 1),
                           (2, 5),
                           (2, 7),
                           (3, 1),
                           (3, 2),
                           (3, 3),
                           (3, 4),
                           (3, 5),
                           (3, 7),
                           (3, 9),
                           (4, 7),
                           (5, 0),
                           (5, 1),
                           (5, 2),
                           (5, 5),
                           (5, 7),
                           (6, 0),
                           (6, 2),
                           (6, 4),
                           (6, 6),
                           (7, 0),
                           (7, 2),
                           (7, 3),
                           (8, 0),
                           (8, 5),
                           (8, 6),
                           (8, 7),
                           (8, 8),
                           (8, 9),
                           (8, 0),
                           (9, 0),
                           (9, 2),
                           (9, 3)]

        # action space
        self.action_space = [0, 1, 2, 3]
        # state space
        self.state_space = [(i, j) for i in range(w) for j in range(h)]

    def reset(self):
        self.state = self.start
        self.done = False

    def move(self, action):
        if action == 0:
            new_state = (self.state[0] + 1, self.state[1])
        elif action == 1:
            new_state = (self.state[0] - 1, self.state[1])
        elif action == 2:
            new_state = (self.state[0], self.state[1] + 1)
        elif action == 3:
            new_state = (self.state[0], self.state[1] - 1)

        if new_state[0] < 0 or \
                new_state[0] >= self.width or \
                new_state[1] < 0 or \
                new_state[1] >= self.height or \
                new_state in self.roadblocks:
            # if the new state is out of bounds, return penalty and current state
            return self.penalty, self.state
        elif new_state == self.goal:
            self.state = new_state
            self.done = True
            return self.reward, self.state
        else:
            self.state = new_state
            return self.penalty, self.state


class MazeAgent:
    def __init__(self, env):
        self.learned_threshold = 0.05
        self.env = env
        self.q = {env.state: np.zeros(shape=(len(env.action_space)))}
        self.epsilon = 0.5
        self.alpha = 0.5

        self.move_count = 0
        self.states_visited = [self.env.start]

        self.learned = False
        self.initial_q = copy.deepcopy(self.q)

    def learning_mode(self):
        self.epsilon = 0.1

    def competition_mode(self):
        self.epsilon = 0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_space)
        else:
            return np.argmax(self.q[state])

    def reset(self):
        self.move_count = 0
        self.states_visited = [self.env.start]
        self.initial_q = copy.deepcopy(self.q)

    def act(self, action):

        self.move_count += 1
        state = self.env.state
        q_state_to_update = self.q[state]

        action = self.choose_action(state)

        reward, new_state = self.env.move(action)

        if new_state not in self.q:
            self.q[new_state] = np.zeros(shape=(len(self.env.action_space)))

        q_state_to_update[action] = (1 - self.alpha) * q_state_to_update[action] + \
                                    self.alpha * (reward + np.max(self.q[new_state]))

        self.states_visited.append(new_state)

        if self.max_update() < self.learned_threshold:
            self.learned = True

    def max_update(self):
        max_update = 0

        default_q = np.zeros(shape=(len(self.env.action_space)))
        for state, q_values in self.q.items():
            q_values = self.q[state]
            old_q_values = self.initial_q.get(state, default_q)
            diff = np.abs(q_values - old_q_values)

            max_update = max(max_update, np.max(diff))

        return max_update

    def summarise(self):
        print("States visited:")
        maze = np.zeros((self.env.width, self.env.height))
        for state in self.states_visited:
            maze[state[0], state[1]] = 1

        for state in self.env.roadblocks:
            maze[state[0], state[1]] = '8'

        print(maze)


if __name__ == "__main__":
    env = MazeEnvironment(10, 10, (0, 0), (9, 9))
    agent = MazeAgent(env)
    agent.learning_mode()

    i = 0
    while not agent.learned:
        i += 1
        while not env.done:
            agent.act(env.state)

        print(f"Learning iteration {i} - Number of moves: {agent.move_count}")
        agent.reset()
        env.reset()

    print()
    print("Evaluation time!")
    agent.competition_mode()
    while not env.done:
        agent.act(env.state)

    print(f"Number of moves: {agent.move_count}")
    agent.summarise()
