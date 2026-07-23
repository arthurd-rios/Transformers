import math
import random
import statistics

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

def relu(v):
    
    # ReLU Logic

    return [max(0, x) for x in v]

def positional_encoding(v, pos):

    n = 10000
    dmodel = len(v)
    i = 0
    even = True

    vpe = []

    for j, emb in enumerate(v):

        if even:

            pe = math.sin(pos/(pow(n, 2*i/dmodel)))

            even = False

        else:

            pe = math.cos(pos/(pow(n, 2*i/dmodel)))

            even = True
            i = i + 1

        vpe.append(emb + pe)        

    return vpe

def initialization_matrix(m, rows, cols, limit):

    for _ in range(rows):

        c = [0.0 for _ in range(cols)]
        m.append(c)

    for i in range(rows):

        for j in range(cols):

            m[i][j] = random.uniform(-limit, limit)

    return m

class Linear:
    
    def __init__(self, dmodel, dvoc):

        self.W = []
        self.dmodel = dmodel
        self.dvoc = dvoc

    def create_layer(self):
        
        x = math.sqrt(6/(self.dmodel + self.dvoc))

        initialization_matrix(self.W, self.dmodel, self.dvoc, x)    

    def process_layer(self, word):
        
        match = matrix_multiply(word, self.W)

        return match

class _HeadAttention:
    
    def __init__(self, dmodel, dmha):

        self.dmodel = dmodel
        self.dmha = dmha
        self.head_matrices = {"Wq" : [], "Wk" : [], "Wv" : []}

    def create_head(self):
        
        x = math.sqrt(3/self.dmodel)

        for m in self.head_matrices:
            
            initialization_matrix(m, self.dmodel, self.dmha, x)

class MultiHeadAttention:
    
    def __init__(self, dmodel, mask=None):

        self.dmodel = dmodel
        self.dmha = 64
        self.heads = []
        self.Wo = []
        self.mask = mask

    def create_layer(self):
        
        for _ in range(8):

            head = _HeadAttention(self.dmodel, self.dmha)
            self.heads.append(head)

        x = math.sqrt(3/self.dmodel)

        initialization_matrix(self.Wo, self.dmha*8, self.dmodel, x)

    def process_layer(self, wordsQ, wordsK, wordsV):
        
        matrices_mha = []

        for head in self.heads:

            Q = matrix_multiply(wordsQ, head["Wq"])
            K = matrix_multiply(wordsK, head["Wk"])
            V = matrix_multiply(wordsV, head["Wv"])

            KT = [list(row) for row in zip(*K)]

            QKT = matrix_multiply(Q, KT)

            div = math.sqrt(self.dmha)

            QKT = [[n/div for n in row] for row in QKT]

            if self.mask:
                
                for i in range(len(QKT)):
                    
                    for j in range(len(QKT[i])):
                        
                        if j > i:

                            QKT[i][j] = -math.inf
                        
                        else:

                            QKT[i][j] = QKT[i][j]

            for row in QKT:

                softmax(row)

            att = matrix_multiply(QKT, V)

            matrices_mha.append(att)

        fmatrix_mha = [sum(rows, []) for rows in zip(*matrices_mha)]

        final_matrix = matrix_multiply(fmatrix_mha, self.Wo)

        return final_matrix

class FeedForward:
    
    def __init__(self, dmodel):

        self.W1 = []
        self.W2 = []
        self.b1 = []
        self.b2 = []
        self.dmodel = dmodel
        self.dff = dmodel * 4

    def create_layer(self):

        x = math.sqrt(6/(self.dmodel + self.dff))

        initialization_matrix(self.W1, self.dmodel, self.dff, x)
        initialization_matrix(self.W2, self.dff, self.dmodel, x)

        self.b1 = [0.0 for _ in range(self.dff)]
        self.b2 = [0.0 for _ in range(self.dff)]

    def process_layer(self, words):

        fwords = []

        for word in words:

            x1 = matrix_multiply(word, self.W1)

            x1 = [emb + bias1 for emb, bias1 in zip(x1, self.b1)]

            x1 = relu(x1)

            x2 = matrix_multiply(x1, self.W2)

            x2 = [emb + bias2 for emb, bias2 in zip(x2, self.b2)]

            fwords.append(x2)

        return fwords

class AddNorm:

    def __init__(self, dmodel):
        
        self.beta = []
        self.gamma = []
        self.dmodel = dmodel

    def create_layer(self):

        self.beta = [0.0 for _ in range(self.dmodel)]
        self.gamma = [1.0 for _ in range(self.dmodel)]

    def _add(self, memb1, memb2):
        
        m = [[memb1[i][j] + memb2[i][j] for j in range (len(memb1[0]))] for i in range(len(memb1))]

        return m

    def _norm(self, memb):
        
        eps = 0.000001

        for i in range(len(memb)):

            row_mean = statistics.mean(memb[i])
            row_pvar = statistics.pvariance(memb[i])

            for j in range(len(memb[i])):

                memb[i][j] = ((self.gamma)[j] * (memb[i][j] - row_mean)/math.sqrt(row_pvar + eps)) + (self.beta)[j]

        return memb

    def process_layer(self, words1, words2):
        
        m = self._add(words1, words2)
        m = self._norm(m)

        return m