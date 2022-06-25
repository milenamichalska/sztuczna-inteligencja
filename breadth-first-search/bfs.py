import numpy as np

class NQueens():
    def __init__(self, N):
        self.N = N
        self.queens_vector = []

    def _generate_children_states(self):
        starting_state = self.queens_vector
        possible_states = []
        for i in range(self.N):
            possible_states.append(starting_state + [i])
        return possible_states

    def solve(self):
        for i in range(n):
            states = self._generate_children_states()
            for state in states:
                conditions_met = True
                if conditions_met:
                    self.queens_vector = state
                    break
        print(self.queens_vector)

n = 4
problem = NQueens(n)
problem.solve()