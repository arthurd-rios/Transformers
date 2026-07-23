import utils

class Decoder:
    
    def __init__(self, dmodel):

        self.dmodel = dmodel
        self.mha1 = utils.MultiHeadAttention(dmodel, True)
        self.addnorm1 = utils.AddNorm(dmodel)
        self.mha2 = utils.MultiHeadAttention(dmodel)
        self.addnorm2 = utils.AddNorm(dmodel)
        self.feedforward = utils.FeedForward(dmodel)
        self.addnorm3 = utils.AddNorm(dmodel)

    def create_decoder(self):

        self.mha1.create_layer()
        self.addnorm1.create_layer()
        self.mha2.create_layer()
        self.addnorm2.create_layer()
        self.feedforward.create_layer()
        self.addnorm3.create_layer()

    def process_decoder(self, words, second_input):

        attention_matrix1 = self.mha1.process_layer(words, words, words)

        addnorm_matrix1 = self.addnorm1.process_layer(attention_matrix1, words)

        attention_matrix2 = self.mha2.process_layer(wordsQ=addnorm_matrix1, wordsK=second_input, wordsV=second_input)

        addnorm_matrix2 = self.addnorm2.process_layer(attention_matrix2, addnorm_matrix1)

        feedforward_matrix = self.feedforward.process_layer(addnorm_matrix2)

        addnorm_matrix3 = self.addnorm3.process_layer(feedforward_matrix, addnorm_matrix2)

        return addnorm_matrix3