import torch
from causal_conv1d import causal_conv1d_fn_hcu
from conv1d_triton_kernel import causal_conv1d_fn
from copy import deepcopy

PAD_SLOT_ID = -1

def test_varlen_fwd_interface():
    x = torch.randn(8058, 8192).float().cuda().transpose(-1, -2)
    weight = torch.randn(8192, 4).float().cuda()
    conv_states = torch.randn(129, 8192, 3).float().cuda()
    query_start_loc = torch.tensor([   0,  995, 2495, 3995, 5508, 7034, 8058]).int().cuda()
    cache_indices = torch.tensor([51, 52, 53, 54, 55, 56]).int().cuda()
    has_initial_state = torch.tensor([True, False, False, False, False, False]).int().cuda().bool()
    # has_initial_state = torch.tensor([False, False, False, False, False, False]).int().cuda().bool()
    seq_lens_cpu = torch.tensor([995, 1500, 1500, 1513, 1526, 1024]).int().tolist()

    conv_states_2 = deepcopy(conv_states)
    assert query_start_loc.ndim == 1

    out_ref = causal_conv1d_fn(
                x,
                weight,
                None,
                activation="silu",
                conv_states=conv_states,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            )
    torch.cuda.synchronize()

    out = causal_conv1d_fn_hcu(
                x,
                weight,
                None,
                conv_states_2,
                query_start_loc,
                cache_indices,
                has_initial_state,
                seq_lens_cpu,
                "silu",
                -1
            )
    torch.cuda.synchronize()

    print("max=", torch.abs(out-out_ref).max())
    print("mean=", torch.abs(out-out_ref).mean())

    rtol, atol = 1e-2, 5e-2
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    import triton
    def benchmark_causal_conv1d():
        _ = causal_conv1d_fn_hcu(
                x,
                weight,
                None,
                conv_states_2,
                query_start_loc,
                cache_indices,
                has_initial_state,
                seq_lens_cpu,
                "silu",
                -1
            )
    t = triton.testing.do_bench_cudagraph(benchmark_causal_conv1d)
    print(f"\x1b[35mPerformance: {t:.3f} ms\x1b[0m")
    print("-----------------------------------------------------------------------------------")

if __name__ == "__main__":
    test_varlen_fwd_interface()