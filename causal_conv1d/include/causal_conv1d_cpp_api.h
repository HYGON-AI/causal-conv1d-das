#pragma once

#include <torch/extension.h>

// Public C++ entry points exported by causal_conv1d_cuda*.so.
//
// These functions launch on at::cuda::getCurrentCUDAStream() / current HIP
// stream and write into caller-provided output tensors. They intentionally
// mirror the existing pybind API in csrc/causal_conv1d.cpp so external C++
// code can link against a wheel-installed causal_conv1d_cuda extension.

void causal_conv1d_fwd(const at::Tensor& x,
                       const at::Tensor& weight,
                       const c10::optional<at::Tensor>& bias,
                       const c10::optional<at::Tensor>& seq_idx,
                       const c10::optional<at::Tensor>& initial_states,
                       at::Tensor& out,
                       c10::optional<at::Tensor>& final_states_out,
                       bool silu_activation);

void causal_conv1d_update(const at::Tensor& x,
                          const at::Tensor& conv_state,
                          const at::Tensor& weight,
                          const c10::optional<at::Tensor>& bias,
                          at::Tensor& out,
                          bool silu_activation,
                          const c10::optional<at::Tensor>& cache_seqlens,
                          const c10::optional<at::Tensor>& conv_state_indices);

void causal_conv1d_fwd_hcu(const at::Tensor& x,
                           const at::Tensor& weight,
                           const c10::optional<at::Tensor>& bias,
                           const c10::optional<at::Tensor>& initial_states,
                           at::Tensor& out,
                           at::Tensor& query_start_loc,
                           c10::optional<at::Tensor>& cache_indices,
                           c10::optional<at::Tensor>& has_initial_state,
                           int max_seqlen,
                           c10::optional<std::string> activation,
                           int pad_slot_id);

namespace causal_conv1d {
namespace cpp {

inline at::Tensor fwd(const at::Tensor& x,
                      const at::Tensor& weight,
                      const c10::optional<at::Tensor>& bias = c10::nullopt,
                      const c10::optional<at::Tensor>& seq_idx = c10::nullopt,
                      const c10::optional<at::Tensor>& initial_states = c10::nullopt,
                      c10::optional<at::Tensor> final_states_out = c10::nullopt,
                      bool silu_activation = false) {
  at::Tensor out = at::empty_like(x);
  causal_conv1d_fwd(x, weight, bias, seq_idx, initial_states, out, final_states_out, silu_activation);
  return out;
}

inline at::Tensor update(const at::Tensor& x,
                         const at::Tensor& conv_state,
                         const at::Tensor& weight,
                         const c10::optional<at::Tensor>& bias = c10::nullopt,
                         bool silu_activation = false,
                         const c10::optional<at::Tensor>& cache_seqlens = c10::nullopt,
                         const c10::optional<at::Tensor>& conv_state_indices = c10::nullopt) {
  at::Tensor out = at::empty_like(x);
  causal_conv1d_update(x, conv_state, weight, bias, out, silu_activation, cache_seqlens, conv_state_indices);
  return out;
}

inline at::Tensor fwd_hcu(const at::Tensor& x,
                          const at::Tensor& weight,
                          const c10::optional<at::Tensor>& bias,
                          const c10::optional<at::Tensor>& initial_states,
                          at::Tensor& query_start_loc,
                          c10::optional<at::Tensor> cache_indices,
                          c10::optional<at::Tensor> has_initial_state,
                          int max_seqlen,
                          c10::optional<std::string> activation = c10::nullopt,
                          int pad_slot_id = -1) {
  at::Tensor out = at::empty_like(x);
  causal_conv1d_fwd_hcu(x,
                        weight,
                        bias,
                        initial_states,
                        out,
                        query_start_loc,
                        cache_indices,
                        has_initial_state,
                        max_seqlen,
                        activation,
                        pad_slot_id);
  return out;
}

}  // namespace cpp
}  // namespace causal_conv1d
