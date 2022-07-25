from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='React_extend',
    ext_modules=[
        CUDAExtension('roi_align.Align1D', [
            'roi_align/src/roi_align_cuda.cpp',
            'roi_align/src/roi_align_kernel.cu']),

        CUDAExtension('grid_sample1d.grid_sample1d_cuda', [
            'grid_sample1d/grid_sample1d_cuda.cpp',
            'grid_sample1d/grid_sample1d_cuda_kernel.cu',
        ]),

    ],

    cmdclass={
        'build_ext': BuildExtension
    })
