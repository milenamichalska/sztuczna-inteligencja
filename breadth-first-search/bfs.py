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
        search_counter = 1
        while(len(states) > 0):
            # print(states[-1])
            search_counter += 1
            if (self._conditions_met(states[-1])):
                return [states[-1], search_counter]
            else:
                children_states = self._generate_children_states(states[-1])
                if (len(children_states[0]) > 4):
                    states = states[:-1]
                else:
                    states = states[:-1] + children_states
        print(self.queens_vector)

n = 4
problem = NQueens(n)
print(problem.solve())