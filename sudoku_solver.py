import numpy as np


# def Crooks_algorithm(matrix):

class back_tracking:
    def __init__(self, matrix):
        self.matrix = matrix

    def solver(self):
        find = self.find_empty(self.matrix)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if back_tracking.check(self.matrix, i, (row, col)):
                self.matrix[row][col] = i

                if back_tracking.solver(self.matrix):
                    return True

                self.matrix[row][col] = 0

        return False

    # check the row, col and squares
    def check(self, num, pos):

        # check the row
        for i in range(len(self.matrix[0])):
            if self.matrix[pos[0]][i] == num and pos[1] != i:
                return False

        # check the col
        for i in range(len(self.matrix)):
            if self.matrix[i][pos[1]] == num and pos[0] != i:
                return False

        # check square
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if self.matrix[i][j] == num and (i, j) != pos:
                    return False

        return True

    def print_matrix(self):
        for i in range(len(self.matrix)):
            if i % 3 == 0 and i != 0:
                print("------------------")

            for j in range(len(self.matrix[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")

                if j == 8:
                    print(self.matrix[i][j])
                else:
                    print(str(self.matrix[i][j]) + " ", end="")

    def find_empty(self):
        
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                if self.matrix[i][j] == 0:
                    return i, j  # row, col

        return None
