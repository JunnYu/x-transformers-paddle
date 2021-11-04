import paddle
from paddle.autograd import PyLayer


def _roll_last(X, axis):
    if axis == -1:
        return X
    elif axis < 0:
        axis = X.ndim - axis

    perm = [i for i in range(X.ndim) if i != axis] + [axis]
    return X.transpose(perm)


def _make_ix_like(X, axis):
    d = X.shape[axis]
    rho = paddle.arange(1, d + 1, dtype=X.dtype)
    view = [1] * X.ndim
    view[0] = -1
    reshaped = rho.reshape(view)
    new_perm = list(range(reshaped.ndim))
    new_perm[-1] = 0
    new_perm[0] = reshaped.ndim - 1
    return reshaped.transpose(new_perm)


def _entmax_threshold_and_support(X, axis=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.

    Parameters
    ----------
    X : paddle.Tensor
        The input tensor to compute thresholds over.

    axis : int
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    tau : paddle.Tensor like `X`, with all but the `axis` dimension intact
        the threshold value for each vector
    support_size : paddle LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """

    if k is None or k >= X.shape[axis]:  # do full sort
        Xsrt = paddle.sort(X, axis=axis, descending=True)
    else:
        Xsrt, _ = paddle.topk(X, k=k, axis=axis)

    rho = _make_ix_like(Xsrt, axis)
    mean = Xsrt.cumsum(axis) / rho
    mean_sq = (Xsrt ** 2).cumsum(axis) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho

    # NOTE this is not exactly the same as in reference algo
    # Fortunately it seems the clamped values never wrongly
    # get selected by tau <= sorted_z. Prove this!
    delta_nz = paddle.clip(delta, 0)
    tau = mean - paddle.sqrt(delta_nz)

    support_size = (tau <= Xsrt).sum(axis, keepdim=True)

    if tau.ndim > 2:
        raw_shape = tau.shape
        tau = tau.flatten(0, -2)
        tau_star = tau.index_sample((support_size - 1).flatten(0, -2)).reshape(
            raw_shape[:-1] + [-1]
        )
    else:
        tau_star = tau.index_sample(support_size - 1)

    if k is not None and k < X.shape[axis]:
        unsolved = (support_size == k).squeeze(axis)

        if paddle.any(unsolved):
            X_ = _roll_last(X, axis)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, axis=-1, k=2 * k)
            _roll_last(tau_star, axis)[unsolved] = tau_
            # cast
            raw_dtype = support_size.dtype
            _roll_last(support_size.astype("float64"), axis)[unsolved] = ss_.astype(
                "float64"
            )
            support_size = support_size.astype(raw_dtype)

    return tau_star, support_size


class Entmax15Function(PyLayer):
    @staticmethod
    def forward(ctx, X, axis=0, k=None):
        max_val = X.max(axis=axis, keepdim=True)
        X = X - max_val  # same numerical stability trick as for softmax
        X = X / 2  # divide by 2 to solve actual Entmax

        tau_star, _ = _entmax_threshold_and_support(X, axis=axis, k=k)

        Y = paddle.clip(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y.detach())
        ctx.axis = axis
        return Y

    @staticmethod
    def backward(ctx, dY):
        (Y,) = ctx.saved_tensor()
        gppr = paddle.sqrt(Y)  # = 1 / g'' (Y)
        dX = dY * gppr
        q = dX.sum(ctx.axis) / gppr.sum(ctx.axis)
        q = q.unsqueeze(ctx.axis)
        return dX - q * gppr


def entmax15(X, axis=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.

    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.

    Parameters
    ----------
    X : paddle.Tensor
        The input tensor.

    axis : int must
        The dimension along which to apply 1.5-entmax.

    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.

    Returns
    -------
    P : paddle tensor, same shape as X
        The projection result, such that P.sum(axis=axis) == 1 elementwise.
    """
    assert axis in [-1, X.ndim - 1]
    return Entmax15Function.apply(X, axis, k)


if __name__ == "__main__":
    # 当前paddle仅支持 axis=-1的条件！
    import torch
    from entmax import entmax15 as pt_entmax15

    k = 3
    x = paddle.randn((3, 4, 12, 33, 5))
    o1 = torch.tensor(entmax15(x, k=k).numpy())
    o2 = pt_entmax15(torch.tensor(x.numpy()), k=k)
    d = (o1 - o2).abs()
    print(d.mean())
    print(d.max())
    # tensor(1.2753e-08)
    # tensor(4.7684e-07)
