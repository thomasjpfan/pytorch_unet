"""Customizable UNet Implementation"""

import torch.nn as nn

from .blocks import (
    SplitBlock, CenterBlock, MergeBlock, FinalBlock
)


class UNet(nn.Module):

    def __init__(self, *,
                 input_shape,
                 num_classes,
                 layers=4,
                 features_root=16,
                 split_block=SplitBlock,
                 merge_block=MergeBlock,
                 center_block=CenterBlock,
                 final_block=FinalBlock,
                 double_center_features=True):
        """Customizable UNet, assumes image is a square

        `split_block` is a `nn.Module` that will be initialized like so:
        ```
        split_block(in_shape, out_shape, hor_shape, layer)
        ```
        where `{}_shape` is a tuple (features, size) and layer is the layer number
        the split_block resides, the `{}_shape` arguments can be used to have sanity checks
        during the initialization of `UNet`. `in_shape` is the shape coming into the block,
        `out_shape` is the expected shape of the downscaled signal, `hor_shape` is the expected
        shape of the signal going to the merge_block in the same layer.

        When `split_block.forward(input)` is called, it must return a tuple,
        (output, horizontal)

        `merge_block` is a `nn.Module` that is initialized like so:
        ```
        merge_block(in_shape, out_shape, hor_shape, layer)
        ```
        exactly the same as `split_block`. `merge_block.forward(input, horizontal)`
        is called, it must return output with `out_shape`

        `center_block` is called after `layers` of `split_block`s are called.
        And is initlized
        like so:
        ```
        center_block(in_feats, out_feats)
        ```
        where `in_feats` is the number of features coming into the block and
        `out_feats` is the number of features coming out. When `double_center_features`
        is True, the out_feats will be double the `in_feats`.

        Args:
            input_shape (tuple): Input shape of image (features, size)
            layers (int): Number of layers in UNet
            num_classes (int): Number of classes to output
            split_block (nn.Module): module that splits input into
                two signals: highway and output, where the output is
                downscaled
            merge_block (nn.Module): module that merges highway and input
                into one signal: output that is upscaled
            center_block (nn.Module): center module
            features_root (int, optional): features to start with in first layer
            double_center_features (bool, optional): center layer should double the number of
                features

        Raises:
            ValueError: input_shape is not compatible

        """
        super().__init__()
        input_feats, input_size = input_shape

        max_layer_scale = 2**layers
        if input_size % max_layer_scale != 0:
            msg = (f"input size: {input_size} not divisible by "
                   f"2**layers: {max_layer_scale}")
            raise ValueError(msg)

        layers_l = list(range(layers))

        def layer_to_in_shape(layer):
            if not layer:
                return input_shape
            return (features_root * 2**(layer-1), input_size // 2**layer)

        def layer_to_out_shape(layer):
            return (features_root * 2**layer, input_size // 2**(layer+1))

        def layer_to_hor_shape(layer):
            return (features_root * 2**layer, input_size // 2**layer)

        in_shapes = map(layer_to_in_shape, layers_l)
        out_shapes = map(layer_to_out_shape, layers_l)
        hor_shapes = map(layer_to_hor_shape, layers_l)

        all_shapes = zip(in_shapes, out_shapes, hor_shapes)

        for layer, (in_shape, out_shape, hor_shape) in enumerate(all_shapes):
            down_name = f"d{layer}"
            self.add_module(
                down_name,
                split_block(in_shape, out_shape, hor_shape, layer)
            )

            # Depends on if features were doubled in the center layer
            in_feat, in_size = out_shape
            if double_center_features:
                in_feat *= 2
            out_feat, out_size = in_feat//2, in_size*2

            merge_in_shape = (in_feat, in_size)
            merge_out_shape = (out_feat, out_size)

            merge_name = f"m{layer}"
            self.add_module(
                merge_name,
                merge_block(merge_in_shape, merge_out_shape, hor_shape, layer)
            )

        in_feats = 2**(layers-1) * features_root
        out_feats = 2*in_feats if double_center_features else in_feats
        self.center = center_block(in_feats, out_feats)

        self.final_feats = features_root if double_center_features else features_root // 2
        self.final = final_block(self.final_feats, num_classes)
        self.layers = layers

    def forward(self, x):
        highway = []

        for i in range(self.layers):
            down_name = f"d{i}"
            x, hw = self.__getattr__(down_name)(x)
            highway.append(hw)

        x = self.center(x)

        for i in reversed(range(self.layers)):
            merge_name = f"m{i}"
            x = self.__getattr__(merge_name)(x, highway[i])

        return self.final(x)
