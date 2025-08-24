#include <torch/serialize/tensor.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>
#include "ballquery_cuda_kernel.h"

// 필요하다면, THCState 관련 extern 선언은 제거할 수 있습니다.
//extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void ballquery_cuda(int b, int n, int m, float radius, int nsample, 
                    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor)
{
    // 최신 API: .data_ptr<T>() 사용
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    ballquery_cuda_launcher(b, n, m, radius, nsample, new_xyz, xyz, idx);
}

void ballquery_cuda_fast(int b, int n, int m, float radius, int nsample, 
                         at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor)
{
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);

    // 최신 API: .data_ptr<T>() 사용
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    int *idx = idx_tensor.data_ptr<int>();

    // 최신 API로 CUDA 스트림 가져오기
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ballquery_cuda_launcher_fast(b, n, m, radius, nsample, new_xyz, xyz, idx, stream);
}
