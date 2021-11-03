import paddle
from paddle.autograd import PyLayer


class EntmaxBisectFunction(PyLayer):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(paddle.clip(X, min=0), alpha)

    @classmethod
    def forward(
        cls,
        ctx,
        X,
        alpha=paddle.to_tensor(1.5),
        axis=-1,
        n_iter=50,
        ensure_sum_one=True,
    ):

        if not isinstance(alpha, paddle.Tensor):
            alpha = paddle.to_tensor(alpha, dtype=X.dtype)

        alpha_shape = X.shape
        alpha_shape[axis] = 1
        alpha = alpha.expand(alpha_shape)

        ctx.alpha = alpha
        ctx.axis = axis
        d = X.shape[axis]

        X = X * (alpha - 1)

        max_val = X.max(axis=axis, keepdim=True)

        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)

        f_lo = cls._p(X - tau_lo, alpha).sum(axis) - 1

        dm = tau_hi - tau_lo

        for it in range(n_iter):

            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(axis) - 1

            mask = (f_m * f_lo >= 0).unsqueeze(axis)
            tau_lo = paddle.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(axis=axis).unsqueeze(axis=axis)

        ctx.save_for_backward(p_m)

        return p_m

    @classmethod
    def backward(cls, ctx, dY):
        (Y,) = ctx.saved_tensors

        gppr = paddle.where(Y > 0, Y ** (2 - ctx.alpha), paddle.zeros((1,)))

        dX = dY * gppr
        q = dX.sum(ctx.axis) / gppr.sum(ctx.axis)
        q = q.unsqueeze(ctx.axis)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = paddle.where(Y > 0, Y * paddle.log(Y), paddle.zeros((1,)))
            # shannon entropy
            ent = S.sum(ctx.axis).unsqueeze(ctx.axis)
            Y_skewed = gppr / gppr.sum(ctx.axis).unsqueeze(ctx.axis)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.axis).unsqueeze(ctx.axis)

        return dX, d_alpha


def entmax_bisect(X, alpha=1.5, axis=-1, n_iter=50, ensure_sum_one=True):
    """alpha-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to both X and alpha.

    Parameters
    ----------
    X : paddle.Tensor
        The input tensor.

    alpha : float or paddle.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != axis else 1)
        A value of alpha=2 corresponds to sparsemax, and alpha=1 corresponds to
        softmax (but computing it this way is likely unstable).

    axis : int
        The dimension along which to apply alpha-entmax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Returns
    -------
    P : paddle tensor, same shape as X
        The projection result, such that P.sum(axis=axis) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, axis, n_iter, ensure_sum_one)


if __name__ == "__main__":
    import torch
    from entmax import entmax_bisect as pt_entmax_bisect

    x = paddle.randn((3, 4, 5, 6))
    o1 = torch.tensor(entmax_bisect(x, axis=-2).numpy())
    o2 = pt_entmax_bisect(torch.tensor(x.numpy()), dim=-2)
    d = (o1 - o2).abs()
    print(d.mean())
    print(d.max())
