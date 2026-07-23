import utils
import encoder
import decoder

class Transformers:
    
    def __init__(self, dmodel):
        
        self.encoders = [encoder.Encoder(dmodel) for _ in range(6)]
        self.decoders = [decoder.Decoder(dmodel) for _ in range (6)]
        self.linear = utils.Linear(dmodel)
        