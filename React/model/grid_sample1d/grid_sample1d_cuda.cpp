#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor grid_sample1d_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    bool padding_mode,
    bool align_corners);

std::vector<torch::Tensor> grid_sample1d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor grid,
    bool padding_mode,
    bool align_corners);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor grid_sample1d_forward(
    torch::Tensor input,
    torch::Tensor grid,
    bool padding_mode,
    bool align_corners) {
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  return grid_sample1d_cuda_forward(input, grid, padding_mode, align_corners);
}

std::vector<torch::Tensor> grid_sample1d_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor grid,
    bool padding_mode,
    bool align_corners) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(input);
  CHECK_INPUT(grid);
  return grid_sample1d_cuda_backward(grad_output,input,grid,padding_mode,align_corners);
//  return grid_sample1d_cuda_backward(
//      grad_output,
//      input,
//      grid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &grid_sample1d_forward, "grid sample forward (CUDA)");
  m.def("backward", &grid_sample1d_backward, "grid sample backward (CUDA)");
}
