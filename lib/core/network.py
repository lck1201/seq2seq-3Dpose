import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn, rnn
from config import config
nJoints = config.NETWORK.nJoints

class MyLSTM(gluon.Block):
    def __init__(self, cfg, **kwargs):
        super(MyLSTM, self).__init__(**kwargs)
        self.hidden_dim = cfg.NETWORK.hidden_dim
        with self.name_scope():
            self.drop1 = nn.Dropout(cfg.NETWORK.dropout1)
            self.drop2 = nn.Dropout(cfg.NETWORK.dropout2)
            self.encoder = rnn.LSTMCell(hidden_size=self.hidden_dim, prefix='encoder_')
            self.decoder = rnn.LSTMCell(hidden_size=self.hidden_dim, prefix='decoder_')
            self.output_layer = nn.Dense(3*nJoints)
            
    def forward(self, inputs, init_state, start_token):
        state = init_state
        for item in inputs:
            mid_hidden, state = self.encoder(self.drop1(item), state)

        # decoder
        ins = start_token
        pred = [] #seqLength, 64x(3x16)
        for i in range(config.DATASET.seqLength):
            hidden_state, state = self.decoder(self.drop1(ins), state)
            output = self.output_layer(self.drop2(hidden_state)) + ins
            ins = output
            pred.append(output)
        return pred

def get_net(cfg):
    return MyLSTM(cfg)