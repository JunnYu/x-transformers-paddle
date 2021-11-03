from math import ceil

import paddle
import paddle.nn.functional as F
from paddle import nn

from pd_x_transformers.entmax_bisect import entmax_bisect


def exists(val):
    return val is not None


pos_nef = 1e4
neg_inf = -pos_nef
# nucleus


def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = paddle.topk(logits, k=logits.shape[-1])
    cum_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = (cum_probs > (1 - thres)).astype("int64")
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_indices = (
        sorted_indices + paddle.arange(logits.shape[0]).unsqueeze(-1) * logits.shape[-1]
    )
    condition = paddle.scatter(
        sorted_indices_to_remove.flatten(),
        sorted_indices.flatten(),
        sorted_indices_to_remove.flatten(),
    )
    condition = condition.astype("bool").reshape(logits.shape)
    logits = paddle.where(condition, paddle.full_like(logits, neg_inf), logits)

    return logits


def top_p_new(logits, thres=0.9):
    sorted_logits = paddle.sort(logits, axis=-1, descending=True)
    cum_probs = paddle.cumsum(F.softmax(sorted_logits, axis=-1), axis=-1)
    cum_probs[:, 1:] = cum_probs[:, :-1].clone()
    cum_probs[:, 0] = 0
    logits_masked = paddle.where(
        cum_probs > (1 - thres),
        paddle.to_tensor(pos_nef, dtype=sorted_logits.dtype),
        sorted_logits,
    )
    min_logits = paddle.min(logits_masked, axis=1, keepdim=True)
    return paddle.where(
        logits >= min_logits, logits, paddle.to_tensor(neg_inf, dtype=logits.dtype)
    )


# topk


def top_k(logits, thres=0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = paddle.topk(logits, k)
    logits = logits.clone()
    logits[logits < val[:, -1].unsqueeze(-1)] = neg_inf
    return logits


# top_a


def top_a(logits, min_p_pow=2.0, min_p_ratio=0.02):
    probs = F.softmax(logits, axis=-1)
    limit = paddle.pow(paddle.max(probs), min_p_pow) * min_p_ratio
    logits[probs < limit] = neg_inf
    logits[probs >= limit] = 1.0
    return logits


# entmax

ENTMAX_ALPHA = 1.3
entmax = entmax_bisect


class AutoregressiveWrapper(nn.Layer):
    def __init__(self, net, ignore_index=-100, pad_value=0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @paddle.no_grad()
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token=None,
        temperature=1.0,
        filter_logits_fn=top_k,
        filter_thres=0.9,
        min_p_pow=2.0,
        min_p_ratio=0.02,
        **kwargs,
    ):
        was_training = self.net.training
        num_dims = start_tokens.ndim

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop("mask", None)

        if mask is None:
            mask = paddle.full_like(out, True, dtype="bool")

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len :]
            mask = mask[:, -self.max_seq_len :]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, axis=-1)

            elif filter_logits_fn is top_a:
                filtered_logits = filter_logits_fn(
                    logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio
                )
                probs = F.softmax(filtered_logits / temperature, axis=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, axis=-1)

            sample = paddle.multinomial(probs, 1)

            out = paddle.concat((out, sample), axis=-1)
            mask = (
                F.pad(
                    mask[None].astype("float64"), (0, 1), value=True, data_format="NCL"
                )
                .squeeze(0)
                .astype("bool")
            )

            if exists(eos_token):
                is_eos_tokens = out == eos_token
                if is_eos_tokens.any(axis=-1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(
                        is_eos_tokens[..., :-1][None], (1, 0), data_format="NCL"
                    ).squeeze(0)
                    mask = shifted_is_eos_tokens.astype("float64").cumsum(axis=-1) >= 1
                    out = paddle.where(
                        mask, paddle.to_tensor(self.pad_value, dtype=mask.dtype), out
                    )
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask

        out = self.net(xi, **kwargs)
        loss = F.cross_entropy(
            out.flatten(0, 1), xo.flatten(), ignore_index=self.ignore_index
        )
        return loss
