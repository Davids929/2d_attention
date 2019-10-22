#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from .attention_cell import MultiHeadAttentionCell, DotProductAttentionCell

class FeedForward(gluon.HybridBlock):
    def __init__(self, hidden_size, output_size, dropout=0.0, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        with self.name_scope():
            self.dense1 = nn.Dense(hidden_size, activation='relu', flatten=False)
            self.dense2 = nn.Dense(output_size, flatten=False)
            self.layer_norm = nn.LayerNorm()
            self.dropout = nn.Dropout(dropout)
    
    def hybrid_forward(self, F, x):
        output = self.dense1(x)
        output = self.dense2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)
        return output

class TransformerEncoderLayer(gluon.HybridBlock):
    def __init__(self, hidden_size=1024, output_size=512, 
                 num_heads=8, dropout=0.0, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout = nn.Dropout(dropout)
            dot_atten_cell_1 = DotProductAttentionCell(dropout=dropout)
            self.multi_head_atten_1 = MultiHeadAttentionCell(dot_atten_cell_1, 
                                    output_size, output_size, output_size, num_heads)
            self.layer_norm1 = nn.LayerNorm()
            self.feed_forward = FeedForward(hidden_size, output_size, dropout=dropout)
    
    def hybrid_forward(self, F, x, mask=None):
        output, att_weight1 = self.multi_head_atten_1(x, x, x, mask)
        output = self.dropout(output)
        output = self.layer_norm1(x + output)
        output = self.feed_forward(output)
        return output, att_weight1

class RelationAttention(gluon.HybridBlock):
    def __init__(self, num_layers=4, hidden_size=1024, output_size=512, 
                 num_heads=8, dropout=0.0, **kwargs):
        super(RelationAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.body  = nn.HybridSequential()
            self.dense = nn.Dense(output_size, flatten=False, use_bias=False)
            for _ in range(num_layers):
                self.body.add(TransformerEncoderLayer(hidden_size=hidden_size,
                                                      output_size=output_size,
                                                      num_heads=num_heads,
                                                      dropout=dropout))
    def hybrid_forward(self, F, x):
        x = self.dense(x)
        for block in self.body:
            output, _ = block(x)
            x = output
        return x

class ParallelAttention(gluon.HybridBlock):
    '''Parameters
    ----------
    hidden_size : int
        The number of units in the hidden layer.
    output_size : int
        The number of output character.
    '''
    def __init__(self, hidden_size=512, output_size=40, **kwargs):
        super(ParallelAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.dense1 = nn.Dense(hidden_size, flatten=False, use_bias=False, 
                                   activation='tanh')
            self.dense2 = nn.Dense(output_size, flatten=False, use_bias=False)
        
    def hybrid_forward(self, F, x, mask):
        '''
        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input feature map. shape:[batch_size, time_step, channel]
        Returns
        -------
        mxnet.nd.NDArray
        '''

        energy = self.dense1(x)
        energy = self.dense2(energy)
        energy = F.broadcast_mul(energy, mask)
        energy = F.broadcast_plus(energy, (1.0 - mask) * (-10000.0))
        energy = F.transpose(energy, axes=(0,2,1))
        att_weight = F.softmax(energy, axis=-1)
        return att_weight

class Encoder(gluon.HybridBlock):
    def __init__(self, backbone, ra_kwargs=None, pa_kwargs=None,
                 rnn_kwargs=None, rnn_type='LSTM', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            self.backbone = backbone
            self.relation_att = RelationAttention(**ra_kwargs)
            self.parallel_att = ParallelAttention(**pa_kwargs)
           
            if rnn_kwargs is not None:
                if rnn_type == 'LSTM':
                    self.rnn = gluon.rnn.LSTM(**rnn_kwargs)
                elif rnn_type == 'GRU':
                    self.rnn = gluon.rnn.GRU(**rnn_kwargs)
                else:
                    self.rnn = None 
                    raise Warning('rnn type is not correct!')
            else:
                self.rnn = None

    def hybrid_forward(self, F, x, mask):

        feat, feat_mask = self.backbone(x, mask)
        inp = F.reshape(feat, shape=(0, 0, -1))
        inp = F.transpose(inp, axes=(0, 2, 1))
        inp_mask = F.reshape(feat_mask, shape=(0, 0, -1))
        inp_mask = F.transpose(inp_mask, axes=(0, 2, 1))
        output = self.relation_att(inp)
        output = F.broadcast_mul(output, inp_mask)
        att_mask = self.parallel_att(output, inp_mask)
        output   = F.batch_dot(att_mask, inp)
        if self.rnn is not None:
            output = self.rnn(output)
        return output, att_mask
        
