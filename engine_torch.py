import torch
import triton

def benchmark(workload:str, batch: int, n: int, m: int, k: int, acc_dtype: str, out_dtype: str, provider: str, out_dir: str):
    if acc_dtype == "f16":
        input_dtype = torch.float16
    a = torch.randn((m, k), device='cuda', dtype=input_dtype)
    b = torch.randn((k, n), device='cuda', dtype=input_dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * batch * m * n * k * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


def bench_torch(args, out_dir):
    torch_workload_dict = {
        "GEMM-1024-1024-1024": [1024, 1024, 1024],
        "GEMM-4096-4096-4096": [4096, 4096, 4096],
    }
    if args.workload[0:4] == "GEMM":
        res, _, _ = benchmark(workload=args.workload,
            batch=args.batch_size, 
            m=torch_workload_dict[args.workload][0],
            n=torch_workload_dict[args.workload][1],
            k=torch_workload_dict[args.workload][2],
            acc_dtype=args.acc_dtype, 
            out_dtype=args.out_dtype,
            provider="torch", 
            out_dir=out_dir,
            )
    else:
        raise Exception("Unsupported operator!")
    print(res)
