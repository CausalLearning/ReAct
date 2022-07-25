#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

static __forceinline__ __device__
bool within_bounds(int h, int H) {
  return h >= 0 && h < H;
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__
scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template<typename scalar_t>
static __forceinline__ __device__
scalar_t safe_downgrade_to_int_range(scalar_t x){
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}


template<typename scalar_t>
static __forceinline__ __device__
scalar_t compute_coordinates(scalar_t coord, int size,
                             bool padding_mode,
                             bool align_corners) {
  if (padding_mode) { // True for border padding
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int size,
    bool padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int size,
                                           bool align_corners, scalar_t *grad_in) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename scalar_t>
static __forceinline__ __device__
scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, scalar_t *grad_in) {
  // Note that it is important for the gradient calculation that borders
  // are considered out of bounds.
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}

template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord,
    int size,
    bool padding_mode,
    bool align_corners,
    scalar_t *grad_in) {
  scalar_t grad_clip, grad_refl;
  coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode) { // true for border padding
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  }
  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
__global__ void grid_sample1d_cuda_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grid,
    scalar_t* __restrict__ output,
    bool padding_mode,
    bool align_corners,
    const int N,
    const int L_in,
    const int batch_size,
    const int C,
    const int L_out) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N){
    const int l = index % L_out;
    const int c = (index/L_out) % C;
    const int n = index / (C * L_out);

    const int grid_offset = n * L_out + l;

    scalar_t x = grid[grid_offset];
    scalar_t ix = grid_sampler_compute_source_index(x, L_in, padding_mode, align_corners);

    const int index_left = ::floor(ix);
    const int index_right = index_left + 1;

//    const int output_offset = l + c * L_out + n * C * L_out;
    const int output_offset = l + c * L_out + n * C * L_out;
    scalar_t surface_left = index_right-ix;
    scalar_t surface_right = ix-index_left;

    const int input_left_offset = index_left + c * L_in + n * L_in * C;
    const int input_right_offset = index_right + c * L_in + n * L_in * C;
    output[output_offset] = static_cast<scalar_t>(0);
    if(within_bounds(index_left, L_in)){
        output[output_offset] += input[input_left_offset] * surface_left;
    }
    if(within_bounds(index_right, L_in)){
        output[output_offset] += input[input_right_offset] * surface_right;
    }
//    output[output_offset] = (ix-index_left) * (input[input_right_offset] - input[input_left_offset]) + input[input_left_offset];
  }
}

template <typename scalar_t>
__global__ void grid_sample1d_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grid,
    scalar_t* __restrict__ grad_input,
    scalar_t* __restrict__ grad_grid,
    bool padding_mode,
    bool align_corners,
    const int N,
    const int L_in,
    const int batch_size,
    const int C,
    const int L_out) {

  const int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N){
    const int l = index % L_out;
    const int n = index / L_out;

    const int grid_offset = n * L_out + l;
//
    scalar_t x = grid[grid_offset];
    scalar_t gix_mult;
    scalar_t ix = grid_sampler_compute_source_index_set_grad(x, L_in, padding_mode, align_corners, &gix_mult);
//
    const int index_left = ::floor(ix);
    const int index_right = index_left + 1;


    scalar_t surface_left = index_right-ix;
    scalar_t surface_right = ix-index_left;

    scalar_t iy = static_cast<scalar_t>(0);
    scalar_t iy_se = static_cast<scalar_t>(1);

    scalar_t gix = static_cast<scalar_t>(0);

    for(int c=0; c<C;++c){
        const int output_offset = l + c * L_out + n * C * L_out;
        const int grad_output_offset = l + c * L_out + n * C * L_out;

        const int input_left_offset = index_left + c * L_in + n * L_in * C;
        const int input_right_offset = index_right + c * L_in + n * L_in * C;

        scalar_t gOut = grad_output[grad_output_offset];

        if (within_bounds(index_left, L_in)) {
            atomicAdd(grad_input + input_left_offset, surface_left * gOut);
        }
        if(within_bounds(index_right, L_in)){
            atomicAdd(grad_input + input_right_offset, surface_right * gOut);
        }

        if (within_bounds(index_left, L_in)) { // order is important
    //        gix -= surface_left * input[input_left_offset] * gOut;
            gix -= input[input_left_offset] * (iy_se-iy) * gOut;
        }

        if(within_bounds(index_right, L_in)){
    //        gix += surface_right * input[input_right_offset] * gOut;
            gix += input[input_right_offset] * (iy_se-iy) * gOut;
        }
    }
    grad_grid[grid_offset] =  gix*gix_mult;
  }
}
}

torch::Tensor grid_sample1d_cuda_forward(
    torch::Tensor input,
    torch::Tensor grid,
    bool padding_mode,
    bool align_corners) {

  const auto batch_size = input.size(0);
  const auto C = input.size(1);
  const auto L_in = input.size(2);

  const auto L_out = grid.size(1);

  torch::Tensor output = torch::zeros({batch_size, C, L_out}, input.options());

  const int threads = 1024;
//  const dim3 blocks((C*L_out + threads - 1) / threads, batch_size);
  const int N = C*L_out*batch_size;
  const int blocks = (N + threads-1)/ threads;

  AT_DISPATCH_FLOATING_TYPES(input.type(), "lltm_forward_cuda", ([&] {
    grid_sample1d_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        grid.data<scalar_t>(),
        output.data<scalar_t>(),
        padding_mode,
        align_corners,
    N,
    L_in,
    batch_size,
    C,
    L_out);
  }));

  return output;
}


std::vector<torch::Tensor> grid_sample1d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor grid,
    bool padding_mode,
    bool align_corners) {

    const auto batch_size = input.size(0);
    const auto C = input.size(1);
    const auto L_in = input.size(2);

    const auto L_out = grid.size(1);

    torch::Tensor grad_input = torch::zeros_like(input);
    torch::Tensor grad_grid = torch::zeros_like(grid);

    const int threads = 1024;
    const int N = L_out*batch_size;
    const int blocks = (N + threads-1)/ threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "grid_sample1d_backward_cuda", ([&] {
    grid_sample1d_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        grad_output.data<scalar_t>(),
        input.data<scalar_t>(),
        grid.data<scalar_t>(),
        grad_input.data<scalar_t>(),
        grad_grid.data<scalar_t>(),
        padding_mode,
        align_corners,
        N,
        L_in,
        batch_size,
        C,
        L_out);
  }));
    return {grad_input, grad_grid};
}