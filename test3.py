import numpy as np

def compute_covariance( x, y):
    print(x.shape)
    print(x)
    print(y.shape)
    matrix_1 = np.concatenate((x, y), axis=1)
    print(matrix_1.shape)
    matrix_ones = np.ones((len(x), 1))
    ones_transpose = matrix_ones.transpose()
    print(matrix_ones.shape)
    matrix_raw = matrix_1 - (len(x)) * ones_transpose.dot(matrix_1)
    matrix_deviation = (1 / len(x))*matrix_raw.transpose().dot(matrix_raw)
    return matrix_deviation


x = np.random.random((300,1))
y =  np.random.random((300,1))
mat = compute_covariance(x,y)
#print (mat)

#print(np.cov(x,y))
matrix_ones = np.ones((76800,1))
print("Ones matrix", matrix_ones)
ones_transpose = matrix_ones.transpose()
print("Will Multiply")
mul_ones = matrix_ones.dot(ones_transpose)
print("Multiplying")
print(mul_ones)