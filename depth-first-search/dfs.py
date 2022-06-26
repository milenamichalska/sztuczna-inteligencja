import numpy as np
import itertools
import matplotlib.pyplot as plt

class NQueens():
    def __init__(self, N):
        self.N = N
        self.queens_vector = []

    def _generate_children_states(self, starting_state):
        possible_states = []
        for i in range(self.N):
            possible_states.append(starting_state + [i])
        return possible_states

    def _conditions_met(self, state):
        conditions_met = True

        if (len(state) < self.N):
            conditions_met = False

        for x, y in itertools.combinations(enumerate(state), 2):
            if (x[0] == y[0] or x[1] == y[1] or (np.abs(x[0] - y[0]) == np.abs(x[1] - y[1]))):
                conditions_met = False

        return conditions_met

    def solve(self):
        states = self._generate_children_states([])
        search_counter = 1
        while(len(states) > 0):
            # print(states[state_i])
            search_counter += 1
            if (self._conditions_met(states[0])):
                return [states[0], search_counter]
            else:
                states = states[1:] + self._generate_children_states(states[0])
        print(self.queens_vector)

# n = 5
# problem = NQueens(n)
# print(problem.solve())

ns = np.arange(4, 9, 1)
state_counters = []

for n in ns:
    problem = NQueens(n)
    _, c = problem.solve()
    state_counters.append(c)

plt.plot(ns, state_counters)
plt.title("Checked states counter dependent on n size")
plt.show()