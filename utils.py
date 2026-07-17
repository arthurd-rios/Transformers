def matrix_multiply(matrix1, matrix2):

    # Get the dimensions of both matrices

    num_rows1 = len(matrix1)
    num_cols1 = len(matrix1[0])

    num_rows2 = len(matrix2)
    num_cols2 = len(matrix2[0])

    # Check if the number of columns in the first matrix equals the number of rows in the second

    if num_cols1 != num_rows2:
        return

    # Create the result matrix

    matrix = [[0 for _ in range(num_cols2)] for _ in range(num_rows1)]

    # Multiply Logical

    for i in range(num_rows1):

        for j in range (num_cols2):

            for k in range (num_cols1):

                matrix[i][j] = matrix[i][j] + matrix1[i][k]*matrix2[k][j] 

    return matrix

def softmax():
    pass

def relu():
    pass

def positional_encoding():
    pass

class Linear:
    pass

class LayerNorm:
    pass

class MultiHeadAttention:
    pass

class FeedForward:
    pass

class AddNorm:
    pass