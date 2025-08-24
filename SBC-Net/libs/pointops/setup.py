from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointops',
    ext_modules=[
        CUDAExtension('pointops_cuda', [
            'src/pointops_api.cpp',
            'src/ballquery/ballquery_cuda.cpp',
            'src/ballquery/ballquery_cuda_kernel.cu',
            'src/knnquery/knnquery_cuda.cpp',
            'src/knnquery/knnquery_cuda_kernel.cu',
            'src/grouping/grouping_cuda.cpp',
            'src/grouping/grouping_cuda_kernel.cu',
            'src/grouping_int/grouping_int_cuda.cpp',
            'src/grouping_int/grouping_int_cuda_kernel.cu',
            'src/interpolation/interpolation_cuda.cpp',
            'src/interpolation/interpolation_cuda_kernel.cu',
            'src/sampling/sampling_cuda.cpp',
            'src/sampling/sampling_cuda_kernel.cu',
            'src/labelstat/labelstat_cuda.cpp',
            'src/labelstat/labelstat_cuda_kernel.cu',
            'src/featuredistribute/featuredistribute_cuda.cpp',
            'src/featuredistribute/featuredistribute_cuda_kernel.cu'
        ],
        extra_compile_args={
            'cxx': ['-g'],
            'nvcc': [
                '-O2',
                '-gencode=arch=compute_89,code=sm_89',
                '-gencode=arch=compute_120,code=sm_120',  # ✅ RTX 5090
                '-Xcompiler', '-fPIC'
            ]
        })
    ],
    cmdclass={'build_ext': BuildExtension}
)
