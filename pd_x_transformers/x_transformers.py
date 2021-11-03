import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

import numpy as np
import paddle
import paddle.nn.functional as F
import torch
from einops import rearrange, repeat


from paddle import einsum, nn
from paddle.nn.initializer import Assign, Constant, KaimingNormal
from paddle.fluid.data_feeder import convert_dtype

# TODO
entmax15 = None


from pd_x_transformers.autoregressive_wrapper import AutoregressiveWrapper


# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple("Intermediates", ["pre_softmax_attn", "post_softmax_attn"])

LayerIntermediates = namedtuple("Intermediates", ["hiddens", "attn_intermediates"])

# helpers


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def max_neg_value(tensor):
    return -paddle.to_tensor(np.finfo(convert_dtype(tensor.dtype)).max)


# init helpers


def init_zero_(layer):
    layer.weight.set_value(paddle.zeros_like(layer.weight))
    if exists(layer.bias):
        layer.bias.set_value(paddle.zeros_like(layer.bias))


# keyword argument helpers


def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(
        partial(string_begins_with, prefix), d
    )
    kwargs_without_prefix = dict(
        map(lambda x: (x[0][len(prefix) :], x[1]), tuple(kwargs_with_prefix.items()))
    )
    return kwargs_without_prefix, kwargs


# activations


class ReluSquared(nn.Layer):
    def forward(self, x):
        return F.relu(x) ** 2


# positional embeddings


class AbsolutePositionalEmbedding(nn.Layer):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        n = paddle.arange(x.shape[1])
        pos_emb = self.emb(n)
        pos_emb = rearrange(pos_emb, "n d -> () n d")
        return pos_emb * self.scale


class FixedPositionalEmbedding(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = paddle.arange(x.shape[seq_dim]).type_as(self.inv_freq) + offset
        sinusoid_inp = paddle.einsum("i , j -> i j", t, self.inv_freq)
        emb = paddle.concat((sinusoid_inp.sin(), sinusoid_inp.cos()), axis=-1)
        return rearrange(emb, "n d -> () n d")


class RelativePositionBias(nn.Layer):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position, causal=True, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).astype("int64") * num_buckets
            n = paddle.abs(n)
        else:
            n = paddle.maximum(n, paddle.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            paddle.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype("int64")
        val_if_large = paddle.min(
            val_if_large, paddle.full_like(val_if_large, num_buckets - 1)
        )

        ret += paddle.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j = qk_dots.shape[-2:]
        q_pos = paddle.arange(i, dtype="int64")
        k_pos = paddle.arange(j, dtype="int64")
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(
            rel_pos,
            causal=self.causal,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return qk_dots + (bias * self.scale)


class AlibiPositionalBias(nn.Layer):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = paddle.to_tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> () h () ()")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                : heads - closest_power_of_2
            ]
        )

    def forward(self, qk_dots):
        h, i, j = qk_dots.shape[-3:]

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias = paddle.arange(j)
        bias = rearrange(bias, "j -> () () () j")
        bias = bias * self.slopes
        # TODO
        bias = F.pad(bias, (0, 0, 0, h - bias.shape[1]), data_format="NHWC")
        self.register_buffer("bias", bias, persistent=False)
        return qk_dots + self.bias


class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads):
        super().__init__(heads)
        self.learned_logslopes = self.create_parameter(
            self.slopes.shape, default_initializer=Assign(paddle.log(self.slopes))
        )

    def forward(self, qk_dots):
        h, i, j = qk_dots.shape[-3:]

        slopes = self.learned_logslopes.exp()
        # TODO
        slopes = F.pad(slopes, (0, 0, 0, h - slopes.shape[1]), data_format="NHWC")

        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :j]
        else:
            bias = paddle.arange(j)
            self.register_buffer("bias", bias, persistent=False)

        return qk_dots + (bias * slopes)


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len):
        t = paddle.arange(max_seq_len).type_as(self.inv_freq)
        freqs = paddle.einsum("i , j -> i j", t, self.inv_freq)
        emb = paddle.concat((freqs, freqs), axis=-1)
        return rearrange(emb, "n d -> () () n d")


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


# norms


class Scale(nn.Layer):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = self.create_parameter(shape=(1,), default_initializer=Constant(1.0))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Layer):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = self.create_parameter(shape=(1,), default_initializer=Constant(1.0))

    def forward(self, x):
        norm = paddle.norm(x, axis=-1, keepdim=True) * self.scale
        return x / norm.clip(min=self.eps) * self.g


class RMSNorm(nn.Layer):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = self.create_parameter(shape=(dim,), default_initializer=Constant(1.0))

    def forward(self, x):
        norm = paddle.norm(x, axis=-1, keepdim=True) * self.scale
        return x / norm.clip(min=self.eps) * self.g


# residual and residual gates


class Residual(nn.Layer):
    def __init__(self, dim, scale_residual=False):
        super().__init__()
        self.residual_scale = (
            self.create_parameter(shape=(dim,), default_initializer=Constant(1.0))
            if scale_residual
            else None
        )

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        return x + residual


class GRUGating(nn.Layer):
    def __init__(self, dim, scale_residual=False):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = (
            self.create_parameter(shape=(dim,), default_initializer=Constant(1.0))
            if scale_residual
            else None
        )

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, "b n d -> (b n) d"), rearrange(residual, "b n d -> (b n) d")
        )

        return gated_output.reshape(x.shape)


# token shifting


def shift(t, amount, mask=None):
    if amount == 0:
        return t

    if exists(mask):
        t = paddle.where(mask[..., None], t, torch.zeros((1,)))

    if t.ndim == 2:
        return F.pad(t[..., :-amount, :][None], (amount, 0), data_format="NLC").squeeze(
            0
        )
    else:
        return F.pad(t[..., :-amount, :], (0, 0, amount, 0), data_format="NCHW")


class ShiftTokens(nn.Layer):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get("mask", None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, axis=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(
            map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts))
        )
        x = paddle.concat((*segments_to_shift, *rest), axis=-1)
        return self.fn(x, **kwargs)


# feedforward


class GLU(nn.Layer):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, axis=-1)
        return x * self.act(gate)


class FeedForward(nn.Layer):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        glu=False,
        relu_squared=False,
        post_act_ln=False,
        dropout=0.0,
        zero_init_output=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        activation = ReluSquared() if relu_squared else nn.GELU()

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), activation)
            if not glu
            else GLU(dim, inner_dim, activation)
        )

        self.net = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.net[-1])

    def forward(self, x):
        return self.net(x)


# attention.


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        dim_head=DEFAULT_DIM_HEAD,
        heads=8,
        causal=False,
        mask=None,
        talking_heads=False,
        head_scale=False,
        collab_heads=False,
        collab_compression=0.3,
        sparse_topk=None,
        use_entmax15=False,
        num_mem_kv=0,
        dropout=0.0,
        on_attn=False,
        gate_values=False,
        zero_init_output=False,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        qk_dim = v_dim = dim_head * heads

        # collaborative heads
        self.collab_heads = collab_heads
        if self.collab_heads:
            qk_dim = int(collab_compression * qk_dim)
            self.collab_mixing = self.create_parameter(
                (heads, qk_dim),
                default_initializer=Assign(paddle.randn((heads, qk_dim))),
            )

        self.to_q = nn.Linear(dim, qk_dim, bias_attr=False)
        self.to_k = nn.Linear(dim, qk_dim, bias_attr=False)
        self.to_v = nn.Linear(dim, v_dim, bias_attr=False)

        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, v_dim)
            self.to_v_gate.weight.set_value(paddle.zeros_like(self.to_v_gate.weight))
            self.to_v_gate.bias.set_value(paddle.ones_like(self.to_v_gate.bias))

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = self.create_parameter(
                (heads, heads), default_initializer=Assign(paddle.randn((heads, heads)))
            )
            self.post_softmax_proj = self.create_parameter(
                (heads, heads), default_initializer=Assign(paddle.randn((heads, heads)))
            )

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = self.create_parameter(
                (1, heads, 1, 1), default_initializer=Constant(1.0)
            )

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = entmax15 if use_entmax15 else F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = self.create_parameter(
                (heads, num_mem_kv, dim_head),
                default_initializer=Assign(paddle.randn((heads, num_mem_kv, dim_head))),
            )
            self.mem_v = self.create_parameter(
                (heads, num_mem_kv, dim_head),
                default_initializer=Assign(paddle.randn((heads, num_mem_kv, dim_head))),
            )

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = (
            nn.Sequential(nn.Linear(v_dim, dim * 2), nn.GLU())
            if on_attn
            else nn.Linear(v_dim, dim)
        )

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        rel_pos=None,
        sinusoidal_emb=None,
        rotary_pos_emb=None,
        prev_attn=None,
        mem=None,
    ):
        b, n, _, h, talking_heads, collab_heads, head_scale, has_context = (
            *x.shape,
            self.heads,
            self.talking_heads,
            self.collab_heads,
            self.head_scale,
            exists(context),
        )
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = paddle.concat((mem, k_input), axis=-2)
            v_input = paddle.concat((mem, v_input), axis=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        if not collab_heads:
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
            )
        else:
            q = einsum("b i d, h d -> b h i d", q, self.collab_mixing)
            k = rearrange(k, "b n d -> b () n d")
            v = rearrange(v, "b n (h d) -> b h n d", h=h)

        if exists(rotary_pos_emb) and not has_context:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(
                lambda t: (t[..., :l], t[..., l:]), (q, k, v)
            )
            ql, kl, vl = map(
                lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl)
            )
            q, k, v = map(
                lambda t: paddle.concat(t, axis=-1), ((ql, qr), (kl, kr), (vl, vr))
            )

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: paddle.ones((b, n)).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: paddle.ones((b, k.shape[-2])).bool())
            q_mask = rearrange(q_mask, "b i -> b () i ()")
            k_mask = rearrange(k_mask, "b j -> b () () j")
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(
                lambda t: repeat(t, "h n d -> b h n d", b=b), (self.mem_k, self.mem_v)
            )
            k = paddle.concat((mem_k, k), axis=-2)
            v = paddle.concat((mem_v, v), axis=-2)
            if exists(input_mask):
                # TODO
                input_mask = F.pad(
                    input_mask.astype("float64"),
                    (self.num_mem_kv, 0, 0, 0),
                    value=1,
                    data_format="NCHW",
                ).astype("bool")

        if collab_heads:
            k = k.expand(shape=[-1, h, -1, -1])

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if talking_heads:
            dots = einsum("b h i j, h k -> b k i j", dots, self.pre_softmax_proj)

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots = paddle.where(input_mask, dots, mask_value)
            del input_mask

        if exists(attn_mask):
            assert (
                2 <= attn_mask.ndim <= 4
            ), "attention mask must have greater than 2 dimensions but less than or equal to 4"
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, "i j -> () () i j")
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, "h i j -> () h i j")
            dots = paddle.where(attn_mask, dots, mask_value)

        if self.causal:
            i, j = dots.shape[-2:]
            r = paddle.arange(i)
            mask = rearrange(r, "i -> () () i ()") < rearrange(r, "j -> () () () j")
            # TODO
            mask = F.pad(
                mask.astype("float64"), (j - i, 0, 0, 0), value=0, data_format="NCHW"
            ).astype(mask.dtype)
            dots = paddle.where(mask, mask_value, dots)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, axis=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots = paddle.where(mask, mask_value, dots)
            del mask

        attn = self.attn_fn(dots, axis=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum("b h i j, h k -> b k i j", attn, self.post_softmax_proj)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        if head_scale:
            out = out * self.head_scale_params

        out = rearrange(out, "b h n d -> b n (h d)")

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * F.sigmoid(gates)

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn, post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates


class AttentionLayers(nn.Layer):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        alibi_learned=False,
        rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        position_infused_attn=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        scale_residual=False,
        shift_tokens=0,
        sandwich_norm=False,
        zero_init_branch_output=False,
        **kwargs,
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.LayerList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = (
            FixedPositionalEmbedding(dim) if position_infused_attn else None
        )

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = (
            RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )

        assert not (
            alibi_pos_bias and rel_pos_bias
        ), "you can only choose Alibi positional bias or T5 relative positional bias, not both"
        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"

        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(
                scale=dim_head ** 0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert (
                alibi_num_heads <= heads
            ), "number of ALiBi heads must be less than the total number of heads"
            assert (
                causal
            ), "ALiBi currently does not work with non-autoregressive mode just yet"
            alibi_pos_klass = (
                LearnedAlibiPositionalBias if alibi_learned else AlibiPositionalBias
            )
            self.rel_pos = alibi_pos_klass(heads=alibi_num_heads)
        else:
            self.rel_pos = None

        assert not (
            not pre_norm and sandwich_norm
        ), "sandwich norm cannot be used when not using prenorm"
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = (
                par_depth * 2 // 3
            )  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert (
                sandwich_coef > 0 and sandwich_coef <= depth
            ), "sandwich coefficient should be less than the depth"
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # iterate and construct layers

        for layer_type, layer_shift_tokens in zip(self.layer_types, shift_tokens):
            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual=scale_residual)

            if sandwich_norm:
                norm = nn.LayerList([norm_fn(), norm_fn()])
            else:
                norm = norm_fn()

            self.layers.append(nn.LayerList([norm, layer, residual]))

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        mems=None,
        return_hiddens=False,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(
                list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems))
            )
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length)

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            if self.sandwich_norm:
                norm, postnorm = norm

            if self.pre_norm:
                x = norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    attn_mask=attn_mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                )
            elif layer_type == "f":
                out = block(x)

            if self.sandwich_norm:
                out = postnorm(out)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c"):
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens, attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on encoder"
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class ViTransformerWrapper(nn.Layer):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers,
        num_classes=None,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), "attention layers must be an Encoder"
        assert (
            image_size % patch_size == 0
        ), "image dimensions must be divisible by the patch size"
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = self.create_parameter(
            (1, num_patches + 1, dim),
            default_initializer=Assign(paddle.randn((1, num_patches + 1, dim))),
        )
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = self.create_parameter(
            (1, 1, dim), default_initializer=Assign(paddle.randn((1, 1, dim)))
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = (
            FeedForward(dim, dim_out=num_classes, dropout=dropout)
            if exists(num_classes)
            else None
        )

    def forward(self, img, return_embeddings=False):
        p = self.patch_size

        x = rearrange(img, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head) or return_embeddings:
            return x

        return self.mlp_head(x[:, 0])


class TransformerWrapper(nn.Layer):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers,
        emb_dim=None,
        max_mem_len=0.0,
        shift_mem_down=0,
        emb_dropout=0.0,
        num_memory_tokens=None,
        tie_embedding=False,
        use_pos_emb=True,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.token_emb = nn.Embedding(num_tokens, emb_dim, weight_attr=KaimingNormal())
        self.pos_emb = (
            AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            if (use_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.to_logits = (
            nn.Linear(dim, num_tokens)
            if not tie_embedding
            else lambda t: t @ self.token_emb.weight  # TODO
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = self.create_parameter(
                (num_memory_tokens, dim),
                default_initializer=Assign(paddle.randn((num_memory_tokens, dim))),
            )

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        **kwargs,
    ):
        b, n, num_mem = *x.shape, self.num_memory_tokens
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = paddle.concat((mem, x), axis=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                # TODO
                if mask.ndim == 2:
                    mask = (
                        F.pad(
                            mask[None].astype("float64"),
                            (num_mem, 0),
                            value=1,
                            data_format="NCL",
                        )
                        .squeeze(0)
                        .astype("bool")
                    )
                else:
                    mask = F.pad(
                        mask, (num_mem, 0, 0, 0), value=1, data_format="NCHW"
                    ).astype("bool")

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[: self.shift_mem_down], mems[self.shift_mem_down :]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)

        if num_mem == 0:
            mem = None
        x = x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(map(lambda pair: paddle.concat(pair, axis=-2), zip(mems, hiddens)))
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems)
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates)
            )
            return out, attn_maps

        return out


class ContinuousTransformerWrapper(nn.Layer):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers,
        dim_in=None,
        dim_out=None,
        emb_dim=None,
        emb_dropout=0.0,
        use_pos_emb=True,
    ):
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.pos_emb = (
            AbsolutePositionalEmbedding(dim, max_seq_len)
            if (use_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
        return_embeddings=False,
        mask=None,
        return_attn=False,
        mems=None,
        **kwargs,
    ):
        b, n, _ = x.shape

        x = self.project_in(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)

        out = self.project_out(x) if not return_embeddings else x

        if return_attn:
            attn_maps = list(
                map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates)
            )
            return out, attn_maps

        return out


class XTransformer(nn.Layer):
    def __init__(self, *, dim, tie_token_emb=False, **kwargs):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim("enc_", kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim("dec_", kwargs)

        assert (
            "dim" not in enc_kwargs and "dim" not in dec_kwargs
        ), "dimension of either encoder or decoder must be set with `dim` keyword"
        enc_transformer_kwargs = pick_and_pop(["num_tokens", "max_seq_len"], enc_kwargs)
        enc_transformer_kwargs["emb_dropout"] = enc_kwargs.pop("emb_dropout", 0)
        enc_transformer_kwargs["num_memory_tokens"] = enc_kwargs.pop(
            "num_memory_tokens", None
        )

        dec_transformer_kwargs = pick_and_pop(["num_tokens", "max_seq_len"], dec_kwargs)
        dec_transformer_kwargs["emb_dropout"] = dec_kwargs.pop("emb_dropout", 0)

        self.encoder = TransformerWrapper(
            **enc_transformer_kwargs, attn_layers=Encoder(dim=dim, **enc_kwargs)
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers=Decoder(dim=dim, cross_attend=True, **dec_kwargs),
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        self.decoder = AutoregressiveWrapper(self.decoder)

    @paddle.no_grad()
    def generate(
        self,
        seq_in,
        seq_out_start,
        seq_len,
        src_mask=None,
        src_attn_mask=None,
        **kwargs,
    ):
        encodings = self.encoder(
            seq_in, mask=src_mask, attn_mask=src_attn_mask, return_embeddings=True
        )
        return self.decoder.generate(
            seq_out_start, seq_len, context=encodings, context_mask=src_mask, **kwargs
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_attn_mask=None):
        enc = self.encoder(
            src, mask=src_mask, attn_mask=src_attn_mask, return_embeddings=True
        )
        out = self.decoder(tgt, context=enc, mask=tgt_mask, context_mask=src_mask)
        return out
