import paddle
import paddle.nn as nn
from paddle.nn.initializer import Uniform

from . import RearrangeMixin, ReduceMixin
from ._einmix import _EinmixMixin

__author__ = "junnyu"


class Rearrange(RearrangeMixin, nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class Reduce(ReduceMixin, nn.Layer):
    def forward(self, input):
        return self._apply_recipe(input)


class EinMix(_EinmixMixin, nn.Layer):
    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        uniform_ = Uniform(-weight_bound, weight_bound)

        self.weight = self.create_parameter(weight_shape, default_initializer=uniform_)

        if bias_shape is not None:
            self.bias = self.create_parameter(bias_shape, default_initializer=uniform_)
        else:
            self.bias = None

    def _create_rearrange_layers(
        self,
        pre_reshape_pattern,
        pre_reshape_lengths,
        post_reshape_pattern,
        post_reshape_lengths,
    ):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **pre_reshape_lengths)

        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(
                post_reshape_pattern, **post_reshape_lengths
            )

    def forward(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = paddle.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result
