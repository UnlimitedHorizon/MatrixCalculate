import numpy as np

class Matrix:
    def __init__(self, output_file):
        super().__init__()
        self.output_file = output_file

    def Transposition(self, array):
        return np.transpose([np.array(array)])

    def SpectralRadius(self, matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return np.max(np.abs(eigenvalues))

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

    def JacobiSolve(self, matrix_A, matrix_b, xs = None, level = 5, need_output = True):
        # if xs is None:
        #     xs = self.Transposition([0]*len(matrix_b))
        matrix_B_j = self.MatrixBJ(matrix_A)
        matrix_f_j = self.MatrixFJ(matrix_A, matrix_b)
        self.output_file.write("GS:\n")
        return self.IterationSolve(matrix_B_j, matrix_f_j, xs, level, need_output)
        # print("J 0: " + str(xs))
        # for i in range(level):
        #     xs = np.dot(matrix_B_j, xs) + matrix_f_j
        #     print("J " + str(i + 1) + str(xs))
        # return xs
    def GaussSeidelSolve(self, matrix_A, matrix_b, xs = None, level = 5, need_output = True):
        # if xs is None:
        #     xs = self.Transposition([0]*len(matrix_b))
        matrix_B_g = self.MatrixBG(matrix_A)
        matrix_f_g = self.MatrixFG(matrix_A, matrix_b)
        self.output_file.write("GS:\n")
        return self.IterationSolve(matrix_B_g, matrix_f_g, xs, level, need_output)
        # print("GS 0: " + str(xs))
        # for i in range(level):
        #     xs = np.dot(matrix_B_g, xs) + matrix_f_g
        #     print("GS " + str(i + 1) + str(xs))
    def SuccessiveOverRelaxation(self, matrix_A, matrix_b, w = 1.0, xs = None, level = 5, need_output = True):
        # if xs is None:
        #     xs = self.Transposition([0]*len(matrix_b))
        matrix_B_sor = self.MatrixBSOR(matrix_A, w)
        matrix_f_sor = self.MatrixFSOR(matrix_A, matrix_b, w)
        self.output_file.write("SOR w = " + str(w) + "\n")
        return self.IterationSolve(matrix_B_sor, matrix_f_sor, xs, level, need_output)
        # print("SOR w = " + str(w))
        # print("SOR 0: " + str(xs))
        # for i in range(level):
        #     xs = np.dot(matrix_B_sor, xs) + matrix_f_sor
        #     print("SOR " + str(i + 1) + str(xs))
    def IterationSolve(self, matrix_B, matrix_f, xs=None, level=5, need_output = True):
        if xs is None:
            xs = self.Transposition([0]*len(matrix_f))
        for i in range(level):
            if need_output:
                self.output_file.write("iteration " + str(i) + ":\n" + str(xs) + "\n")
            xs = np.dot(matrix_B, xs) + matrix_f
        if need_output:
            self.output_file.write("iteration " + str(level) + ":\n" + str(xs) + "\n")
        return xs


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

def Hilbert(n):
    matrix_hilbert = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            matrix_hilbert[i][j] = 1/(i+j+1)
    return matrix_hilbert



f = open("output.txt", "w")
f2 = open("output2.txt", "w")
m = Matrix(f)

# m_A = np.array(augmented_matrix_exercise_2_1)[:, :-1]
# m_b = m.Transposition(np.array(augmented_matrix_exercise_2_1)[:, -1])

# wb = 1.012823
# matrix_B_sor = m.MatrixBSOR(m_A, wb)
# matrix_f_sor = m.MatrixFSOR(m_A, m_b, wb)
# print(matrix_B_sor)
# print(matrix_f_sor)
# m.SuccessiveOverRelaxation(m_A, m_b, wb, level=5)

n = 6
level = 100
matrix_hilbert = Hilbert(n)
matrix_b = matrix_hilbert.dot(m.Transposition(np.ones(n)))

f2.write("Hilbert n = {}\n".format(n))

f2.write("\nJacobi Solve:\n")
# f2.write("B = {}\n".format(m.MatrixBJ(matrix_hilbert)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBJ(matrix_hilbert))))

w = 1.0
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

w = 1.25
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

w = 1.5
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))


n = 8
matrix_hilbert = Hilbert(n)
matrix_b = matrix_hilbert.dot(m.Transposition(np.ones(n)))
f2.write("\n\nHilbert n = {}\n".format(n))

f2.write("\nJacobi Solve:\n")
# f2.write("B = {}\n".format(m.MatrixBJ(matrix_hilbert)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBJ(matrix_hilbert))))

w = 1.0
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

w = 1.25
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

w = 1.5
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))


n = 10
matrix_hilbert = Hilbert(n)
matrix_b = matrix_hilbert.dot(m.Transposition(np.ones(n)))
f2.write("\n\nHilbert n = {}\n".format(n))

f2.write("\nJacobi Solve:\n")
# f2.write("B = {}\n".format(m.MatrixBJ(matrix_hilbert)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBJ(matrix_hilbert))))

w = 1.0
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

w = 1.25
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

w = 1.5
f2.write("\nSuccessive Over Relaxation Solve:\n")
f2.write("w = {}\n".format(w))
# f2.write("B = {}\n".format(m.MatrixBSOR(matrix_hilbert, w=w)))
f2.write("Spectral Radius = {}\n".format(m.SpectralRadius(m.MatrixBSOR(matrix_hilbert, w=w))))
f2.write("level = {}\n".format(level))
result = m.SuccessiveOverRelaxation(matrix_hilbert, matrix_b, level=level, w=w)
f2.write("result = {}\n".format(result))
f2.write("e = {}\n".format(np.linalg.norm(result - m.Transposition(np.ones(n)))))

print("finish")