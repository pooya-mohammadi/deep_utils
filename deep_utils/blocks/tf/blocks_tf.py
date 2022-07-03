from collections import OrderedDict

from deep_utils.utils.list_utils import shift_lst


class BlocksTF:
    @staticmethod
    def conv_norm_act(
            out_c,
            k=(3, 3),
            s=(1, 1),
            p="valid",
            norm="bn",
            act="relu",
            conv=True,
            index=0,
            conv_kwargs: dict = None,
            norm_kwargs: dict = None,
            act_kwargs: dict = None,
            pooling=None,
            pooling_k=(2, 2),
            pooling_s=(2, 2),
            pool_kwargs=None,
            move_forward=0,
    ):
        from tensorflow import keras

        conv_kwargs = dict() if conv_kwargs is None else conv_kwargs
        norm_kwargs = dict() if norm_kwargs is None else norm_kwargs
        act_kwargs = dict() if act_kwargs is None else act_kwargs
        pool_kwargs = dict() if pool_kwargs is None else pool_kwargs
        k = (k, k) if isinstance(k, int) else k
        s = (s, s) if isinstance(s, int) else s
        p = (p, p) if isinstance(p, int) else p
        pooling_s = (pooling_s, pooling_s) if isinstance(
            pooling_s, int) else pooling_s
        pooling_k = (pooling_k, pooling_k) if isinstance(
            pooling_k, int) else pooling_k
        modules, names = [], []
        if conv:
            modules.append(BlocksTF.conv_2d(out_c, k, s, p, **conv_kwargs))
            names.append(f"conv2d_{index}_f{out_c}_k{k[0]}_s{s[0]}")
        if norm:
            modules.append(BlocksTF.load_layer_norm(norm, **norm_kwargs))
            names.append(f"{norm}2d_{index}")
        if act:
            modules.append(BlocksTF.load_activation(act, **act_kwargs))
            names.append(f"{act}_{index}")
        modules = shift_lst(modules, move_forward=move_forward % 3)
        names = shift_lst(names, move_forward=move_forward % 3)
        if pooling:
            pooling_layer = BlocksTF.load_pooling(
                pooling, pooling_k, pooling_s, **pool_kwargs
            )
            modules.append(pooling_layer)
            names.append(f"{pooling}2d_{index}")
        layer_dict = OrderedDict(
            {name: module for name, module in zip(names, modules)})
        cnn = keras.Sequential()
        for value in layer_dict.values():
            cnn.add(value)
        return cnn

    @staticmethod
    def load_pooling(pooling, k, s, **pooling_kwargs):
        from tensorflow.keras import layers

        if pooling == "max":
            pooling = layers.MaxPooling2D(
                pool_size=k, strides=s, **pooling_kwargs)
        elif pooling == "avg":
            pooling = layers.AveragePooling2D(
                pool_size=k, strides=s, **pooling_kwargs)
        else:
            raise ValueError(
                f"requested pooling {pooling} is not supported. Supported poolings: max, avg"
            )
        return pooling

    @staticmethod
    def load_activation(act, **act_kwargs):
        from tensorflow.keras import layers

        if act == "relu":
            activation = layers.ReLU(**act_kwargs)
        elif act == "leaky_relu":
            activation = layers.LeakyReLU(**act_kwargs)
        else:
            raise ValueError(
                f"requested activation {act} is not supported. Supported activations: relu, leaky_relu"
            )
        return activation

    @staticmethod
    def conv_2d(out_c, k=(3, 3), s=(1, 1), padding="valid", bias=True):
        from tensorflow.keras import layers

        return layers.Conv2D(
            filters=out_c, kernel_size=k, strides=s, padding=padding, use_bias=bias
        )

    @staticmethod
    def load_layer_norm(norm, **norm_kwargs):
        from tensorflow.keras import layers

        if norm == "bn":
            layer_norm = layers.BatchNormalization(**norm_kwargs)
        elif norm == "ln":
            layer_norm = layers.LayerNormalization(**norm_kwargs)
        else:
            raise ValueError(
                f"requested layer_norm {norm} is not supported. Supported layer_norms: bn(batch normalization), "
                f"gn(group normalization"
            )
        return layer_norm

    @staticmethod
    def res_basic_block(in_c, out_c, bias, down_sample=False):
        from tensorflow import keras

        modules = []
        names = []
        for i in range(2):
            modules.append(
                BlocksTF.conv_norm_act(
                    int(in_c / 2) if down_sample and i == 0 else in_c,
                    out_c,
                    act=False if i == 1 else "relu",
                    s=(2, 2) if down_sample and i == 0 else (1, 1),
                    conv_kwargs={"bias": bias},
                )
            )
            names.append(f"cnn_{i}_in{in_c}_f{out_c}")
        res_block = keras.Sequential(
            OrderedDict({name: module for name, module in zip(names, modules)})
        )
        return res_block


if __name__ == "__main__":
    a = BlocksTF.conv_norm_act(32, 64, move_forward=2)
    a.build((1, None, None, 3))
    print(a.summary())
