import math

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange

from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
from causal_conv1d.causal_conv1d_interface import causal_conv1d_update, causal_conv1d_update_ref
from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states, causal_conv1d_varlen_states_ref


@pytest.mark.parametrize("return_final_states", [False, True])
# @pytest.mark.parametrize("return_final_states", [True])
@pytest.mark.parametrize("has_initial_states", [False, True])
# @pytest.mark.parametrize("has_initial_states", [False])
@pytest.mark.parametrize("channel_last", [False, True])
# @pytest.mark.parametrize('channel_last', [True])
@pytest.mark.parametrize("itype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('itype', [torch.float16])
@pytest.mark.parametrize("silu_activation", [False, True])
# @pytest.mark.parametrize('silu_activation', [True])
@pytest.mark.parametrize("has_bias", [False, True])
# @pytest.mark.parametrize('has_bias', [True])
@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize('seqlen', [128])
@pytest.mark.parametrize('dim', [64, 4096 + 32])
def test_causal_conv1d(dim, seqlen, width, has_bias, silu_activation, itype, channel_last, has_initial_states, return_final_states):
    ##################### show params ####################
    print("dim={}, seqlen={}, width={}, has_bias={}, silu_activation={}, itype={}, channel_last={}, has_initial_states={}, return_final_states={}".format(dim, seqlen, width, has_bias, silu_activation, itype, channel_last, has_initial_states, return_final_states))
    ################## test correctness ##################
    if not channel_last and (has_initial_states or return_final_states):
        pytest.skip("Only channel_last support initial_states or return_final_states")
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch = 2
    # batch = 1
    if not channel_last:
        x = torch.randn(batch, 4096 + dim + 64, seqlen, device=device, dtype=itype)[:, 4096:4096 + dim, :].requires_grad_()
    else:
        x = rearrange(
            torch.randn(batch, seqlen, 4096 + dim + 64, device=device, dtype=itype)[:, :, 4096:4096 + dim], "b s d -> b d s"
        ).requires_grad_()
    weight = torch.randn(dim, width, device=device, dtype=torch.float32, requires_grad=True)
    if has_bias:
        bias = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        bias = None
    if has_initial_states:
        initial_states = torch.randn(batch, width - 1, dim, device=device, dtype=itype).transpose(1, 2).requires_grad_()
    else:
        initial_states = None
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    initial_states_ref = initial_states.detach().clone().requires_grad_() if initial_states is not None else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_fn(x, weight, bias, initial_states=initial_states, return_final_states=return_final_states,
                           activation=activation)
    out_ref = causal_conv1d_ref(x_ref, weight_ref, bias_ref, initial_states=initial_states_ref, return_final_states=return_final_states, activation=activation)
    if return_final_states:
        out, final_states = out
        out_ref, final_states_ref = out_ref
        print(f"Final states max diff: {(final_states - final_states_ref).abs().max().item()}")
        print(f"Final states mean diff: {(final_states - final_states_ref).abs().mean().item()}")
        assert torch.allclose(final_states, final_states_ref, rtol=rtol, atol=atol)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    if return_final_states:
        out += F.sigmoid(final_states).sum(dim=-1, keepdim=True)
        out_ref += F.sigmoid(final_states_ref).sum(dim=-1, keepdim=True)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dweight max diff: {(weight.grad - weight_ref.grad).abs().max().item()}")
    if has_bias:
        print(f"dbias max diff: {(bias.grad - bias_ref.grad).abs().max().item()}")
    if has_initial_states:
        print(f"dinitial_states max diff: {(initial_states.grad - initial_states_ref.grad).abs().max().item()}")

    assert torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)
    assert torch.allclose(weight.grad, weight_ref.grad, rtol=rtolw, atol=atolw)
    if has_bias:
        assert torch.allclose(bias.grad, bias_ref.grad, rtol=rtolw, atol=atolw)
    if has_initial_states:
        assert torch.allclose(initial_states.grad, initial_states_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)

    ################## test performance ##################
    x_bench = x.detach()
    weight_bench = weight.detach()
    bias_bench = bias.detach() if bias is not None else None
    initial_states_bench = initial_states.detach() if initial_states is not None else None

    import triton
    def benchmark_causal_conv1d():
        with torch.no_grad():
            _ = causal_conv1d_fn(x_bench, weight_bench, bias_bench, initial_states=initial_states_bench, return_final_states=return_final_states, activation=activation)
    t = triton.testing.do_bench_cudagraph(benchmark_causal_conv1d)
    print(f"\x1b[35mPerformance: {t:.3f} ms \x1b[0m")
    print("-----------------------------------------------------------------------------------")
