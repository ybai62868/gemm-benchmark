# from utils.util import lazy_import
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    A_ptr = A_ptr + (offs_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    B_ptr = B_ptr + (offs_b * stride_bb + offs_k[:, None] * stride_bk  + offs_n[None, :] * stride_bn)
    
    # initialize and iteratively update accumulator
   
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):

        a = tl.load(A_ptr)
        b = tl.load(B_ptr)
        
        acc += tl.dot(a, b)
    
        A_ptr += BLOCK_SIZE_K * stride_ak
        B_ptr += BLOCK_SIZE_K * stride_bk
        
    c = acc.to(tl.float16)
    C_ptr = C_ptr + (offs_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr, c, mask=c_mask)
     
    
    

# we can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=None):
    # checks constraints
    assert len(a.shape) == len(b.shape) == 3
    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    B, M, K = a.shape
    B, K, N = b.shape
    assert (
        K % 32 == 0
    ), "We don't check memory-out-of-bounds with K so K must be divisible by BLOCK_SIZE_K"
    # allocates output
    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
       triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B, 1
    )
    matmul_kernel[grid](
        a, b, c,
        B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        ACTIVATION=activation,
    )
    return c



def init_input(batch, m, n, dtype, acc_dtype):
    # Use small range of values to prevent numerical issues.
    min_exp = -4 if acc_dtype == "float16" else -10
    exponents = torch.randint(min_exp, 0, size=(batch, m, n))
    ret = (2.**exponents).to(getattr(torch, dtype)).to("cuda")
    return ret


def test_correctness(a, b):
    torch_output = torch.bmm(a, b)
    triton_output = matmul(a, b)
    # print(f"triton_output={triton_output}")
    # print(f"torch_output={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")



def benchmark(workload:str, 
              m:int,
              n:int,
              k:int,
              BSA:int, 
              BSB:int, 
              TransA:str, 
              TransB:str, 
              acc_dtype: str, 
              out_dtype: str, 
              provider: str, 
              out_dir: str):
    input_dtype = torch.float16
    torch.manual_seed(0)
    BSC = max(BSA, BSB)    
    quantiles = [0.5, 0.2, 0.8]

    # a = init_input(BSC, m, k, "torch.float16", acc_dtype)
    # b = init_input(BSC, k, n, "torch.float16", acc_dtype)

    a = torch.randn((BSA, m, k), device='cuda', dtype=input_dtype)
    b = torch.randn((BSB, k, n), device='cuda', dtype=input_dtype)
    if TransA == "N" and TransB == "N":
        a = a
        b = b
        test_correctness(a, b)


    elif TransA == "N" and TransB == "T":
        a = a
        b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
        test_correctness(a, b)

    
    elif TransA == "T" and TransB == "N":
        a = a.transpose(-1, -2).contiguous().transpose(-1, -2)
        b = b
        test_correctness(a, b)


    else:
        a = a.transpose(-1, -2).contiguous().transpose(-1, -2)
        b = b.transpose(-1, -2).contiguous().transpose(-1, -2)
        test_correctness(a, b)


    
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)








    
    # input_dtype = torch.float16
    # torch.manual_seed(0)
    # a = torch.randn((BSA, m, k), device='cuda', dtype=input_dtype)
    # b = torch.randn((BSB, k, n), device='cuda', dtype=input_dtype)
    # test_correctness(a, b)
    # BSC = max(BSA, BSB)    
    # quantiles = [0.5, 0.2, 0.8]
    # if provider == 'triton':
    #     ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    # perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
    # return perf(ms), perf(max_ms), perf(min_ms)











def bench_triton(args, out_dir):
    if args.workload[0:4] == "GEMM":
        res, _, _ = benchmark(
            workload=args.workload,
            m=args.m,
            n=args.n,
            k=args.k,
            BSA=args.BSA,
            BSB=args.BSB,
            TransA=args.TransA,
            TransB=args.TransB,
            acc_dtype=args.acc_dtype, 
            out_dtype=args.out_dtype,
            provider="triton", 
            out_dir=out_dir,
            )
    else:
        raise Exception("Unsupported operator!")
    print(res, "TFLOPS")

