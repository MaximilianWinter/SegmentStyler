import torch


class MaskBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors # note: this comma is relevant, as ctx.saved_tensors is a tuple
        return grad_output * mask, torch.zeros_like(mask)