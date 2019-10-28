import numpy as np

class Matrix:
    def __init__(self):
        super().__init__()

    def Transposition(self, array):
        return np.transpose([np.array(array)])

    def SpectralRadius(self, matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return np.max(eigenvalues)

    def Norm(self, matrix, value = 2):
        if value == 1:
            return np.max(np.sum(np.abs(matrix), axis = 0))            
        elif value == 2:
            m = np.array(matrix)
            m = m.T.dot(matrix)
            return (self.SpectralRadius(m))**0.5
        else:
            return np.max(np.sum(np.abs(matrix), axis = 1))

    def MatrixM(self, matrix):
        matrix_m = np.diag(np.diag(matrix))
        return matrix_m
    def MatrixD(self, matrix):
        matrix_d = self.MatrixM(matrix)
        return matrix_d
    def MatrixN(self, matrix):
        matrix_m = self.MatrixM(matrix)
        matrix_n = matrix_m - matrix
        return matrix_n
    def MatrixL(self, matrix):
        matrix_l = -np.tril(matrix, -1)
        return matrix_l
    def MatrixU(self, matrix):
        matrix_u = -np.triu(matrix, 1)
        return matrix_u
    def MatrixBJ(self, matrix):
        matrix_m = self.MatrixM(matrix)
        matrix_n = self.MatrixN(matrix)
        matrix_B_j = np.dot(np.linalg.inv(matrix_m), matrix_n)
        return matrix_B_j
    def MatrixBG(self, matrix):
        matrix_d = self.MatrixD(matrix)
        matrix_l = self.MatrixL(matrix)
        matrix_u = self.MatrixU(matrix)
        matrix_B_g = np.dot(np.linalg.inv(matrix_d - matrix_l), matrix_u)
        return matrix_B_g
    def MatrixFJ(self, matrix, b):
        matrix_m = self.MatrixM(matrix)
        matrix_f_j = np.dot(np.linalg.inv(matrix_m), b)
        return matrix_f_j
    def MatrixFG(self, matrix, b):
        matrix_d = self.MatrixD(matrix)
        matrix_l = self.MatrixL(matrix)
        matrix_f_g = np.dot(np.linalg.inv(matrix_d - matrix_l), b)
        return matrix_f_g
    def MatrixBSOR(self, matrix, w = 1.0):
        matrix_d = self.MatrixD(matrix)
        matrix_l = self.MatrixL(matrix)
        matrix_u = self.MatrixU(matrix)
        matrix_1 = (matrix_d - w*matrix_l)/w
        matrix_2 = ((1-w)*matrix_d + w*matrix_u)/w
        matrix_B_sor = np.dot(np.linalg.inv(matrix_1), matrix_2)
        return matrix_B_sor
    def MatrixFSOR(self, matrix, b, w = 1.0):
        matrix_d = self.MatrixD(matrix)
        matrix_l = self.MatrixL(matrix)
        matrix_u = self.MatrixU(matrix)
        matrix_1 = (matrix_d - w*matrix_l)/w
        matrix_2 = ((1-w)*matrix_d + w*matrix_u)/w
        matrix_f_sor = np.dot(np.linalg.inv(matrix_1), b)
        return matrix_f_sor

    def CoefficientWB(self, matrix):
        matrix_B_g = self.MatrixBG(matrix)
        rg = self.SpectralRadius(matrix_B_g)
        wb = 2/(1+(1-rg)**0.5)
        return wb

    def JacobiSolve(self, matrix_A, matrix_b, xs = None, level = 5):
        if xs is None:
            xs = self.Transposition([0]*len(matrix_b))
        matrix_B_j = self.MatrixBJ(matrix_A)
        matrix_f_j = self.MatrixFJ(matrix_A, matrix_b)
        print("J 0: " + str(xs))
        for i in range(level):
            xs = np.dot(matrix_B_j, xs) + matrix_f_j
            print("J " + str(i + 1) + str(xs))
    def GaussSeidelSolve(self, matrix_A, matrix_b, xs = None, level = 5):
        if xs is None:
            xs = self.Transposition([0]*len(matrix_b))
        matrix_B_g = self.MatrixBG(matrix_A)
        matrix_f_g = self.MatrixFG(matrix_A, matrix_b)
        print("GS 0: " + str(xs))
        for i in range(level):
            xs = np.dot(matrix_B_g, xs) + matrix_f_g
            print("GS " + str(i + 1) + str(xs))
    def SuccessiveOverRelaxation(self, matrix_A, matrix_b, w = 1.0, xs = None, level = 5):
        if xs is None:
            xs = self.Transposition([0]*len(matrix_b))
        matrix_B_sor = self.MatrixBSOR(matrix_A, w)
        matrix_f_sor = self.MatrixFSOR(matrix_A, matrix_b, w)
        print("SOR w = " + str(w))
        print("SOR 0: " + str(xs))
        for i in range(level):
            xs = np.dot(matrix_B_sor, xs) + matrix_f_sor
            print("SOR " + str(i + 1) + str(xs))


augmented_matrix_exercise_2_1 = [
    [10, -1, 0, 9], 
    [-1, 10, -2, 7], 
    [0, -2, 10, 6]]

augmented_matrix_eg_3_1_3 = [
    [10, 3, 1, 14],
    [2, -10, 3, -5],
    [1, 3, 10, 14]]
    
augmented_matrix_eg_3_3_1 = [
    [4, 3, 0, 24],
    [3, 4, -1, 30],
    [0, -1, 4, -24]]

coefficient_matrix_exercise_3_1 = [
    [1, 2, -2],
    [1, 1, 1],
    [2, 2, 1]]


m = Matrix()

m_A = np.array(augmented_matrix_exercise_2_1)[:, :-1]
m_b = m.Transposition(np.array(augmented_matrix_exercise_2_1)[:, -1])

wb = 1.012823
matrix_B_sor = m.MatrixBSOR(m_A, wb)
matrix_f_sor = m.MatrixFSOR(m_A, m_b, wb)
print(matrix_B_sor)
print(matrix_f_sor)
m.SuccessiveOverRelaxation(m_A, m_b, wb, level=5)

# matrix_b_j = m.MatrixBJ(coefficient_matrix_exercise_3_1)
# print("B_J = ", str(m.MatrixBJ(coefficient_matrix_exercise_3_1)))
# print(m.SpectralRadius(matrix_b_j))
# matrix_b_g = m.MatrixBG(coefficient_matrix_exercise_3_1)
# print("B_G = ", str(m.MatrixBG(coefficient_matrix_exercise_3_1)))
# print(m.SpectralRadius(matrix_b_g))



# print(m.MatrixBG(np.array(augmented_matrix_exercise_2_1)[:, :-1]))
# print(m.MatrixBJ(np.array(augmented_matrix_exercise_2_1)[:, :-1]))
# print(m.SpectralRadius(m.MatrixBJ(np.array(augmented_matrix_exercise_2_1)[:, :-1])))
# print(m.CoefficientWB(np.array(augmented_matrix_exercise_2_1)[:, :-1]))


# m.JacobiSolve(m_A, m_b)
# m.GaussSeidelSolve(m_A, m_b)
# m.SuccessiveOverRelaxation(m_A, m_b, 1.25, m.Transposition([1]*3),7)
# m.SuccessiveOverRelaxation(m_A, m_b, 1, m.Transposition([1]*3),7)


# class Solve:
#     def __init__(self, augmented_matrix):
#         super().__init__()
#         self.__solve(augmented_matrix)

#     def __solve(self, augmented_matrix):
#         self.augmented_matrix = np.array(augmented_matrix, dtype=float)
#         (self.row, self.column) = self.augmented_matrix.shape
#         assert (self.row + 1 == self.column)
#         self.__iteration_matrix = np.array(self.augmented_matrix)
#         for i in range(self.row):
#             aii = self.__iteration_matrix[i][i]
#             assert (aii != 0)
#             self.__iteration_matrix[i] = self.__iteration_matrix[i] / (-aii)
#             self.__iteration_matrix[i][i] = 0
#         self.__iteration_matrix[:, -1] *= -1

#     def CoefficientMatrix(self):
#         return np.delete(self.augmented_matrix, -1, axis = 1)

#     def IterationMatrix(self):
#         return np.delete(self.__iteration_matrix, -1, axis = 1)

#     def BArray(self):
#         return self.augmented_matrix[:, -1]
#     def FArray(self):
#         return self.__iteration_matrix[:, -1]

#     def JacobiSolve(self, xs, level = 5):
#         xs = np.transpose([np.array(xs, dtype=float)])
#         m = Matrix()
#         matrix_B_j = m.MatrixBJ(self.augmented_matrix[:, :-1])
#         matrix_f_j = m.MatrixFJ(self.augmented_matrix[:, :-1], np.transpose([self.augmented_matrix[:, -1]]))
#         print("J 0: " + str(xs))
#         for i in range(level):
#             xs = np.dot(matrix_B_j, xs) + matrix_f_j
#             print("J " + str(i + 1) + str(xs))

#     def GaussSeidelSolve(self, xs, level = 5):
#         xs = np.transpose([np.array(xs+[1], dtype=float)])
#         print("GS 0: " + str(xs[:-1]))
#         for i in range(level):
#             for j in range (self.row):
#                 xs[j] = self.__iteration_matrix[j].dot(xs)
#             print("GS " + str(i + 1) + ": " + str(xs[:-1]))

#     def SuccessiveOverRelaxation(self, xs, w, level = 5):
#         xs = np.array(xs + [1], dtype=float)
#         print("SOR w = " + str(w))
#         print("SOR 0: " + str(xs[:-1]))
#         for i in range(level):
#             for j in range (self.row):
#                 xs[j] = (1-w)*xs[j] + w*self.__iteration_matrix[j].dot(xs)
#             print("SOR " + str(i + 1) + ": " + str(xs[:-1]))