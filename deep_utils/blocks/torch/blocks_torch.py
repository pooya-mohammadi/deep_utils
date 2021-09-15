from deep_utils.utils.utils.main import shift_lst
from collections import OrderedDict


class BlocksTorch:
    @staticmethod
    def conv_norm_act(in_c, out_c, k=(3, 3), s=(1, 1), p=(1, 1),
                      norm='bn', act='relu', conv=True, index=0, conv_kwargs: dict = None,
                      norm_kwargs: dict = None, act_kwargs: dict = None,
                      pooling=None, pooling_k=(2, 2), pooling_s=(2, 2),
                      pool_kwargs=None, move_forward=0):
        from torch import nn
        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        norm_kwargs = dict() if norm_kwargs is None else norm_kwargs
        act_kwargs = dict() if act_kwargs is None else act_kwargs
        pool_kwargs = dict() if pool_kwargs is None else pool_kwargs
        k = (k, k) if type(k) is int else k
        s = (s, s) if type(s) is int else s
        p = (p, p) if type(p) is int else p
        pooling_s = (pooling_s, pooling_s) if type(pooling_s) is int else pooling_s
        pooling_k = (pooling_k, pooling_k) if type(pooling_k) is int else pooling_k
        modules, names = [], []
        if conv:
            modules.append(BlocksTorch.conv_2d(in_c, out_c, k, s, p, **conv_kwargs))
            names.append(f"conv2d_{index}_in{in_c}_f{out_c}_k{k[0]}_s{s[0]}")
        if norm:
            modules.append(BlocksTorch.load_layer_norm(norm, out_c=out_c, **norm_kwargs))
            names.append(f"{norm}2d_{index}")
        if act:
            modules.append(BlocksTorch.load_activation(act, **act_kwargs))
            names.append(f"{act}_{index}")
        modules = shift_lst(modules, move_forward=move_forward % 3)
        names = shift_lst(names, move_forward=move_forward % 3)
        if pooling:
            pooling_layer = BlocksTorch.load_pooling(pooling, pooling_k, pooling_s, **pool_kwargs)
            modules.append(pooling_layer)
            names.append(f"{pooling}2d_{index}")
        cnn = nn.Sequential(OrderedDict({name: module for name, module in zip(names, modules)}))
        return cnn

    @staticmethod
    def load_pooling(pooling, k, s, **pooling_kwargs):
        from torch import nn
        if pooling == 'max':
            pooling = nn.MaxPool2d(k, s, **pooling_kwargs)
        elif pooling == 'avg':
            pooling = nn.AvgPool2d(k, s, **pooling_kwargs)
        else:
            raise ValueError(f"requested pooling {pooling} is not supported. Supported poolings: max, avg")
        return pooling

    @staticmethod
    def load_activation(act, **act_kwargs):
        from torch import nn
        if act == 'relu':
            activation = nn.ReLU(**act_kwargs)
        elif act == 'leaky_relu':
            activation = nn.LeakyReLU(**act_kwargs)
        else:
            raise ValueError(f"requested activation {act} is not supported. Supported activations: relu, leaky_relu")
        return activation

    @staticmethod
    def conv_2d(in_c, out_c, k=(3, 3), s=(1, 1), p=(1, 1), groups=1, bias=True, padding_mode='zeros'):
        import torch
        return torch.nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=bias, padding_mode=padding_mode)

    @staticmethod
    def load_layer_norm(norm, out_c, **norm_kwargs):
        from torch import nn
        if norm == 'bn':
            layer_norm = nn.BatchNorm2d(num_features=out_c, **norm_kwargs)
        elif norm == 'gn':
            layer_norm = nn.GroupNorm(num_channels=out_c, **norm_kwargs)
        else:
            raise ValueError(
                f"requested layer_norm {norm} is not supported. Supported layer_norms: bn(batch normalization), "
                f"gn(group normalization")
        return layer_norm

    @staticmethod
    def res_basic_block(in_c, out_c, bias, down_sample=False):
        from torch import nn
        modules = []
        names = []
        for i in range(2):
            modules.append(
                BlocksTorch.conv_norm_act(int(in_c / 2) if down_sample and i == 0 else in_c, out_c,
                                          act=False if i == 1 else 'relu',
                                          s=(2, 2) if down_sample and i == 0 else (1, 1), conv_kwargs={'bias': bias}))
            names.append(f"cnn_{i}_in{in_c}_f{out_c}")
        res_block = nn.Sequential(OrderedDict({name: module for name, module in zip(names, modules)}))
        return res_block


if __name__ == '__main__':
    a = BlocksTorch.conv_norm_act(32, 64, move_forward=2)
    print(a)
