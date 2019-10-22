#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import HybridConcurrent, Identity

class ResidualBlock(gluon.HybridBlock):
    def __init__(self, channels, strides, shortcut_1x1=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(channels, 3, strides=strides, padding=1))
            self.conv.add(nn.BatchNorm())
            self.conv.add(nn.Activation('relu'))
            self.conv.add(nn.Conv2D(channels, 3, strides=1, padding=1))
            if shortcut_1x1:
                self.downsample = nn.HybridSequential()
                self.downsample.add(nn.Conv2D(channels, 1, strides, use_bias=False))
                self.downsample.add(nn.BatchNorm())
            else:
                self.downsample = None
            self.relu = nn.Activation('relu') 

    def hybrid_forward(self, F, x):
        residual = x
        x = self.conv(x)
        if self.downsample:
            residual = self.downsample(residual)
        x = self.relu(x + residual)
        return x

class ConvLayer(gluon.HybridBlock):
    def __init__(self, num_blocks, channels, strides, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        with self.name_scope():
            self.layer = nn.HybridSequential()
            for i in range(num_blocks):
                if i == 0:
                    self.layer.add(ResidualBlock(channels, strides, shortcut_1x1=True))
                else:
                    self.layer.add(ResidualBlock(channels, 1))
            self.layer.add(nn.Conv2D(channels, 3, 1, padding=1))
            self.layer.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        x = self.layer(x)
        return x

class ResNet(gluon.HybridBlock):
    def __init__(self, num_blocks_list=None, channels_list=None, 
                 strides_list=None, num_classes=10, use_backbone=False, 
                 **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.use_backbone = use_backbone
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.body = nn.HybridSequential()
            self.pool = nn.HybridSequential()
            self.conv.add(nn.Conv2D(64, 3, 1, padding=1))
            self.conv.add(nn.Activation('relu'))
            for n, c, s in zip(num_blocks_list, channels_list, strides_list):
                self.body.add(ConvLayer(n, c, s))
                self.pool.add(nn.MaxPool2D((3, 3), strides=s, padding=1))
            if not use_backbone:
                self.fc = nn.HybridSequential()
                self.fc.add(nn.AvgPool2D())
                self.fc.add(nn.Dense(num_classes, use_bias=False))
    
    def hybrid_forward(self, F, x, mask):
        x = self.conv(x)
        x = F.broadcast_mul(x, mask)
        for layer, pool in zip(self.body, self.pool):
            x = layer(x)
            mask = pool(mask)
            x = F.broadcast_mul(x, mask)
        if not self.use_backbone:
            x = self.fc(x)
        return x, mask

class DenseLayer(gluon.HybridBlock):
    def __init__(self, bn_size, growth_rate, dropout, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)

        self.layer = nn.HybridSequential()
        self.out   = HybridConcurrent(axis=1)
        with self.name_scope():
            self.layer.add(nn.BatchNorm())
            self.layer.add(nn.Activation('relu'))
            self.layer.add(nn.Conv2D(bn_size * growth_rate, kernel_size=1, use_bias=False))
            self.layer.add(nn.BatchNorm())
            self.layer.add(nn.Activation('relu'))
            self.layer.add(nn.Conv2D(growth_rate, kernel_size=3, padding=1, use_bias=False))
            if dropout:
                self.layer.add(nn.Dropout(dropout))
            self.out.add(Identity())
            self.out.add(self.layer)

    def hybrid_forward(self, F, x):
        return self.out(x)

class DenseBlock(gluon.HybridBlock):
    def __init__(self, num_layers, bn_size, growth_rate, dropout, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.block = nn.HybridSequential()
        with self.name_scope():
            for _ in range(num_layers):
                self.block.add(DenseLayer(bn_size, growth_rate, dropout))

    def hybrid_forward(self, F, x, mask):
        x = self.block(x)
        x = F.broadcast_mul(x, mask)
        return x, mask

class Transition(gluon.HybridBlock):
    def __init__(self, output_dims, strides=(2, 2), **kwargs):
        super(Transition, self).__init__(**kwargs)
        with self.name_scope():
            self.out = nn.HybridSequential()
            self.pool = nn.MaxPool2D((3,3), strides=strides, padding=1)
            self.out.add(nn.BatchNorm())
            self.out.add(nn.Activation('relu'))
            self.out.add(nn.Conv2D(output_dims, 1, strides=strides, use_bias=False))

    def hybrid_forward(self, F, x, mask):
        x = self.out(x)
        mask = self.pool(mask)
        x = F.broadcast_mul(x, mask)
        return x, mask

class DenseNet(gluon.HybridBlock):
    def __init__(self, init_channels, growth_rate, num_layers_list, strides_list, 
                 bn_size=4, dropout=0, num_classes=1000, use_backbone=True, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        
        with self.name_scope():
            self.first_layer = nn.HybridSequential()
            self.features    = nn.HybridSequential()
            self.first_layer.add(nn.Conv2D(init_channels, kernel_size=7,
                                        strides=2, padding=3, use_bias=False))
            self.first_layer.add(nn.BatchNorm())
            self.first_layer.add(nn.Activation('relu'))
            self.first_pool = nn.MaxPool2D((2, 2), (2, 2))
            
            num_channels = init_channels
            for i, num_layers in enumerate(num_layers_list):
                self.features.add(DenseBlock(num_layers, bn_size, growth_rate, dropout))
                num_channels = num_channels + num_layers * growth_rate
                if i != len(num_layers_list) - 1:
                    num_channels = num_channels // 2
                    self.features.add(Transition(num_channels, strides_list[i]))
            if not use_backbone:
                self.output = nn.HybridSequential()
                self.output.add(nn.BatchNorm())
                self.output.add(nn.Activation('relu'))
                self.output.add(nn.GlobalAvgPool2D())
                self.output.add(nn.Dense(num_classes))
            else:
                self.output = None

    def hybrid_forward(self, F, x, mask):
        out  = self.first_layer(x)
        mask = self.first_pool(mask)
        out  = F.broadcast_mul(out, mask)
        for block in self.features._children.values():
            out, mask = block(out, mask)
        if self.output:
            out = self.output(out)
        return out, mask