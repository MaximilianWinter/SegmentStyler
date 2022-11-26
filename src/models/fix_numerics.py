import torch


class NumericsBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return torch.round(grad_output, decimals=4)

class NumericsForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input, decimals=4)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output