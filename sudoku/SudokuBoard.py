# 1 2 3 | 4 5 6 | 7 8 9
# 1 2 3 | 4 5 6 | 7 8 9
# 1 2 3 | 4 5 6 | 7 8 9
# ------+-------+------
# 1 2 3 | 4 5 6 | 7 8 9
# 1 2 3 | 4 5 6 | 7 8 9
# 1 2 3 | 4 5 6 | 7 8 9
# ------+-------+------
# 1 2 3 | 4 5 6 | 7 8 9
# 1 2 3 | 4 5 6 | 7 8 9
# 1 2 3 | 4 5 6 | 7 8 9

class SudokuBoard:
    def __init__(self, dimension=3):
        self.dimension = dimension
        self.empty()

    def empty(self):
        self.board = [[[0] * self.dimension] * self.dimension]

    def fromString(self, numbers):
        return 0
    #     numArray = list(numbers.replace("|", "").replace("------+-------+------", "").split(" "))
    #     numSquares = []

    #     for x in range(self.dimension):
    #         for y in range(self.dimension):
    #             numSquares.push(numArray[])
