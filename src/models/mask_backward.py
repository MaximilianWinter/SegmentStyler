import torch


class MaskBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask):
        ctx.save_for_backward(input, mask)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        return grad_output * torch.ones_like(input) * mask, torch.zeros_like(mask)