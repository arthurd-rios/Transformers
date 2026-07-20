import math

def matrix_multiply(m1, m2):

    # Get the dimensions of both matrices

    num_rows1 = len(m1)
    num_cols1 = len(m1[0])

    num_rows2 = len(m2)
    num_cols2 = len(m2[0])

    # Check if the number of columns in the first matrix equals the number of rows in the second

    if num_cols1 != num_rows2:
        return

    # Create the result matrix

    matrix = [[0 for _ in range(num_cols2)] for _ in range(num_rows1)]

    # Multiply Logic

    for i in range(num_rows1):

        for j in range (num_cols2):

            for k in range (num_cols1):

                matrix[i][j] = matrix[i][j] + m1[i][k]*m2[k][j] 

    return matrix

def softmax(v):
    
    max_value = max(v)

    # Subtract max value from each number in the list to avoid overflow with big numbers

    v = [n - max_value for n in v]

    expo = [math.exp(n) for n in v]

    total_sum = sum(expo)

    # Softmax Logic

    v = [n / total_sum for n in expo]

    return v

def relu(x):
    
    # ReLU Logic

    return max(0, x)

def positional_encoding(v, pos):

    n = 10000
    dmodel = len(v)
    i = 0
    even = True

    for j, emb in enumerate(v):

        if even:

            pe = math.sin(pos/(pow(n, 2*i/dmodel)))

            even = False

        else:

            pe = math.cos(pos/(pow(n, 2*i/dmodel)))

            even = True
            i = i + 1

        v[j] = emb + pe        

    return v 

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