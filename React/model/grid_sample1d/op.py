import math
from torch import nn
from torch.autograd import Function
import torch
from torch.utils.cpp_extension import load

# build by python setup.py install
import React.model.grid_sample1d.grid_sample1d_cuda as grid_sample1d

# jit
# grid_sample1d = load(
#     'grid_sample1d_cuda', ['React/grid_sample1d/grid_sample1d_cuda.cpp', 'React/grid_sample1d/grid_sample1d_cuda_kernel.cu'], verbose=True)


class GridSample1dFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, padding_mode, align_corners):
        outputs = grid_sample1d.forward(input, grid, padding_mode, align_corners)
        # print(print(outputs))
        ctx.save_for_backward(*(input, grid))
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        outputs = grid_sample1d.backward(grad_output.contiguous(), *ctx.saved_variables, ctx.padding_mode,
                                         ctx.align_corners)
        # outputs = lltm_cuda.backward(
        #     grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_variables)
        # d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        # return d_input, d_weights, d_bias, d_old_h, d_old_cell
        d_input, d_grid = outputs
        # print(d_input)
        # print(d_grid)
        return d_input, d_grid, None, None


class GridSample1d(nn.Module):
    def __init__(self, padding_mode, align_corners):
        '''
        :param padding_mode: True for border padding, False for zero padding
        :param align_corners: same with grid_sample in pytorch
        '''
        super(GridSample1d, self).__init__()
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, input, grid):
        return GridSample1dFunction.apply(input, grid, self.padding_mode, self.align_corners)
