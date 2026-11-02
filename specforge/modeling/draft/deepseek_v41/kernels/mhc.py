"""TileKernels mHC forward/backward bindings, preserving source/destination order."""

import torch


class SplitMixes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, bias, h, eps):
        from tile_kernels.mhc import mhc_pre_split_mixes_fwd

        pre = mixes.new_empty(mixes.shape[0], h)
        post = torch.empty_like(pre)
        comb = mixes.new_empty(mixes.shape[0], h * h)
        mhc_pre_split_mixes_fwd(mixes, scale, bias, pre, post, comb, 2.0, eps)
        ctx.save_for_backward(mixes, scale, bias, post)
        return pre, post, comb.view(-1, h, h)

    @staticmethod
    def backward(ctx, dpre, dpost, dcomb):
        from tile_kernels.mhc import mhc_pre_split_mixes_bwd

        mixes, scale, bias, post = ctx.saved_tensors
        dm, ds, db = (torch.empty_like(t) for t in (mixes, scale, bias))
        mhc_pre_split_mixes_bwd(
            dpre.contiguous(),
            dpost.contiguous(),
            dcomb.flatten(1).contiguous(),
            mixes,
            post,
            scale,
            bias,
            dm,
            ds,
            db,
            2.0,
        )
        return dm, ds, db, None, None


class Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, iterations, eps):
        from tile_kernels.mhc import mhc_sinkhorn_fwd

        out = torch.empty_like(logits)
        mhc_sinkhorn_fwd(logits, out, iterations, eps)
        ctx.save_for_backward(logits)
        ctx.iterations, ctx.eps = iterations, eps
        return out

    @staticmethod
    def backward(ctx, dy):
        from tile_kernels.mhc import mhc_sinkhorn_bwd

        (logits,) = ctx.saved_tensors
        dx = torch.empty_like(logits)
        mhc_sinkhorn_bwd(dy.contiguous(), logits, dx, ctx.iterations, ctx.eps)
        return dx, None, None


class PreMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pre):
        from tile_kernels.mhc import mhc_pre_apply_mix_fwd

        x, pre = x.contiguous(), pre.contiguous()
        out = x.new_empty(x.shape[0], x.shape[-1])
        mhc_pre_apply_mix_fwd(x, pre, out)
        ctx.save_for_backward(x, pre)
        return out

    @staticmethod
    def backward(ctx, dy):
        from tile_kernels.mhc import mhc_pre_apply_mix_bwd

        x, pre = ctx.saved_tensors
        # Upstream backward ADDS into this buffer, rather than overwriting it.
        dx = torch.zeros_like(x)
        dp = mhc_pre_apply_mix_bwd(dy.contiguous(), x, pre, dx)
        return dx, dp


class PostMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, post, comb):
        from tile_kernels.mhc import mhc_post_fwd

        x, residual, post, comb = (t.contiguous() for t in (x, residual, post, comb))
        ctx.save_for_backward(x, residual, post, comb)
        return mhc_post_fwd(x, residual, post, comb)

    @staticmethod
    def backward(ctx, dy):
        from tile_kernels.mhc import mhc_post_bwd

        return mhc_post_bwd(*ctx.saved_tensors, dy.contiguous())
