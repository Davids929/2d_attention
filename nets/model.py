#codinf=utf-8
import mxnet as mx
from mxnet import gluon
from .encoder import Encoder, RelationAttention
from .base_net import ResNet, DenseNet
model_zoo = {'resnet':ResNet,
             'densenet':DenseNet}

class Decoder(gluon.HybridBlock):
    def __init__(self, voc_size, ra_kwargs=None, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():
            self.decode1 = gluon.nn.Dense(voc_size, flatten=False)
            self.relation_att = RelationAttention(**ra_kwargs)
            self.decode2 = gluon.nn.Dense(voc_size, flatten=False)

    def hybrid_forward(self, F, x):
        pred1   = self.decode1(x)
        att_out = self.relation_att(x)
        pred2   = self.decode2(att_out)
        return pred1, pred2

class AttentionModel(gluon.HybridBlock):
    def __init__(self, backbone_kwargs, encoder_kwargs, decoder_kwargs, **kwargs):
        super(AttentionModel, self).__init__(**kwargs)
        backbone_type = backbone_kwargs.pop('backbone_type')
        backbone_fn   = model_zoo[backbone_type] 
        with self.name_scope():
            self.backbone = backbone_fn(**backbone_kwargs)
            self.encoder  = Encoder(self.backbone, **encoder_kwargs)
            self.decoder  = Decoder(**decoder_kwargs)

    def hybrid_forward(self, F, x, mask):
        out, atten_mask = self.encoder(x, mask)
        pred1, pred2 = self.decoder(out)
        return pred1, pred2, atten_mask

    def load_parameters(self, filename, ctx=None, allow_missing=False,
                        ignore_extra=False, load_params=''):
        """Load parameters from file previously saved by `save_parameters`.

        Parameters
        ----------
        filename : str
            Path to parameter file.
        ctx : Context or list of Context, default cpu()
            Context(s) to initialize loaded parameters on.
        allow_missing : bool, default False
            Whether to silently skip loading parameters not represents in the file.
        ignore_extra : bool, default False
            Whether to silently ignore parameters from the file that are not
            present in this Block.

        References
        ----------
        `Saving and Loading Gluon Models \
        <https://mxnet.incubator.apache.org/tutorials/gluon/save_load_params.html>`_
        """
        loaded = mx.nd.load(filename)
        params = self._collect_params_with_prefix()
        if not loaded and not params:
            return

        if not any('.' in i for i in loaded.keys()):
            # legacy loading
            del loaded
            self.collect_params().load(
                filename, ctx, allow_missing, ignore_extra, self.prefix)
            return

        if not allow_missing:
            for name in params.keys():
                assert name in loaded, \
                    "Parameter '%s' is missing in file '%s', which contains parameters: %s. " \
                    "Set allow_missing=True to ignore missing parameters."%(
                        name, filename, _brief_print_list(loaded.keys()))
        for name in loaded:
            if not ignore_extra and name not in params:
                raise ValueError(
                    "Parameter '%s' loaded from file '%s' is not present in ParameterDict, " \
                    "which contains parameters %s. Set ignore_extra=True to ignore. "%(
                        name, filename, _brief_print_list(self._params.keys())))
            if name in params:
                if load_params in name:
                    params[name]._load_init(loaded[name], ctx)

def _brief_print_list(lst, limit=7):
    """Print at most `limit` elements of list."""
    lst = list(lst)
    if len(lst) > limit:
        return _brief_print_list(lst[:limit//2], limit) + ', ..., ' + \
            _brief_print_list(lst[-limit//2:], limit)
    return ', '.join(["'%s'"%str(i) for i in lst])