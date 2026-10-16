import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_update_ref
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states, causal_conv1d_varlen_states_ref

@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("has_cache_seqlens", [False, True])
# @pytest.mark.parametrize('has_cache_seqlens', [True])
@pytest.mark.parametrize("seqlen", [1, 4, 5])
# @pytest.mark.parametrize('seqlen', [4])
@pytest.mark.parametrize("width", [2, 3, 4])
# @pytest.mark.parametrize('width', [4])
@pytest.mark.parametrize("dim", [2048, 2048 + 16, 4096])
# @pytest.mark.parametrize("dim", [2048])
def test_causal_conv1d_update(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype):
    ##################### show params ####################
    print("dim={}, width={}, seqlen={}, has_cache_seqlens={}, has_bias={}, silu_activation={}, itype={}".format(dim, width, seqlen, has_cache_seqlens, has_bias, silu_activation, itype))
    ################## test correctness ##################
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 64
    # batch = 1
    # dim = 64
    x = torch.randn(batch, seqlen, dim, device=device, dtype=itype).transpose(-1, -2)
    state_len = torch.randint(width - 1, width + 10, (1,)).item()
    conv_state = torch.randn(batch, state_len, dim, device=device, dtype=itype).transpose(-1, -2)
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    conv_state_ref = conv_state.detach().clone()
    activation = None if not silu_activation else "silu"
    cache_seqlens = (torch.randint(0, 1024, (batch,), dtype=torch.int32, device=device)
                     if has_cache_seqlens else None)
    out = causal_conv1d_update(x, conv_state, weight, bias, activation=activation, cache_seqlens=cache_seqlens)
    out_ref = causal_conv1d_update_ref(x, conv_state_ref, weight, bias, activation=activation, cache_seqlens=cache_seqlens)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.equal(conv_state, conv_state_ref)
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    ################## test performance ##################
    import triton
    def benchmark_causal_conv1d():
        with torch.no_grad():
            _ = causal_conv1d_update(x, conv_state, weight, bias, activation=activation, cache_seqlens=cache_seqlens)
    t = triton.testing.do_bench_cudagraph(benchmark_causal_conv1d)
    print(f"\x1b[35mPerformance: {t:.3f} ms \x1b[0m")
    print("-----------------------------------------------------------------------------------")
