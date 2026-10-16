#include <iostream>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <causal_conv1d_cpp_api.h>

int main() {
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA/HIP device is not available" << std::endl;
    return 2;
  }

  const int batch = 2;
  const int dim = 8;
  const int seqlen = 16;
  const int width = 3;

  auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16);
  auto x = torch::randn({batch, dim, seqlen}, opts).contiguous();
  auto weight = torch::randn({dim, width}, opts).contiguous();
  auto bias = torch::randn({dim}, opts).contiguous();

  auto out = causal_conv1d::cpp::fwd(
      x,
      weight,
      bias,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      false);

  torch::cuda::synchronize();

  std::cout << "causal_conv1d_fwd C++ call OK" << std::endl;
  std::cout << "out sizes: " << out.sizes() << std::endl;
  std::cout << "out dtype: " << out.dtype() << std::endl;
  return 0;
}
