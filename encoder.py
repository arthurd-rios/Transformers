import utils

class Encoder:

    def __init__(self, dmodel):
        
        self.dmodel = dmodel
        self.mha = utils.MultiHeadAttention(dmodel)
        self.addnorm1 = utils.AddNorm(dmodel)
        self.feedforward = utils.FeedForward(dmodel)
        self.addnorm2 = utils.AddNorm(dmodel)

    def create_encoder(self):
        
        self.mha.create_layer()
        self.addnorm1.create_layer()
        self.feedforward.create_layer()
        self.addnorm2.create_layer()

    def process_encoder(self, words):
        
        attention_matrix = self.mha.process_layer(words, words, words)

        addnorm_matrix1 = self.addnorm1.process_layer(attention_matrix, words)

        feedforward_matrix = self.feedforward.process_layer(addnorm_matrix1)

        addnorm_matrix2 = self.addnorm2.process_layer(feedforward_matrix, addnorm_matrix1)

        return addnorm_matrix2

        