import numpy as np
import itertools

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

        if (len(state) < 4):
            conditions_met = False

        for x, y in itertools.combinations(enumerate(state), 2):
            if (x[0] == y[0] or x[1] == y[1] or (np.abs(x[0] - y[0]) == np.abs(x[1] - y[1]))):
                conditions_met = False

        return conditions_met

    def solve(self):
        states = self._generate_children_states([])
        state_i = 0
        while(len(states) > 0):
            # print(states[state_i])
            if (self._conditions_met(states[state_i])):
                return states[state_i]
            else:
                states = states[1:] + self._generate_children_states(states[state_i])
        print(self.queens_vector)

n = 4
problem = NQueens(n)
print(problem.solve())