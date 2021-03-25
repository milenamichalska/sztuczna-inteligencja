import pytest

from SudokuBoard import SudokuBoard

class TestSudokuBoard:
    def setup_method(self):
        self.board = SudokuBoard()

    def test_sudoku_board_from_string(self):
        string_board = """
        1 2 3 | 4 5 6 | 7 8 9
        1 2 3 | 4 5 6 | 7 8 9
        1 2 3 | 4 5 6 | 7 8 9
        ------+-------+------
        1 2 3 | 4 5 6 | 7 8 9
        1 2 3 | 4 5 6 | 7 8 9
        1 2 3 | 4 5 6 | 7 8 9
        ------+-------+------
        1 2 3 | 4 5 6 | 7 8 9
        1 2 3 | 4 5 6 | 7 8 9
        1 2 3 | 4 5 6 | 7 8 9
        """

        result = self.board.fromString(string_board)
        assert result == [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]

