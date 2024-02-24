import torch
import triton


def benchmark(workload:str, BSA:int, BSB:int, n: int, m: int, k: int, acc_dtype: str, out_dtype: str, provider: str, out_dir: str):
    input_dtype = torch.float16
    torch.manual_seed(0)
    a = torch.randn((BSA, m, k), device='cuda', dtype=input_dtype)
    b = torch.randn((BSB, k, n), device='cuda', dtype=input_dtype)
    BSC = max(BSA, BSB)    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

def bench_torch(args, out_dir):
    if args.workload[0:4] == "GEMM":
        res, _, _ = benchmark(
            workload=args.workload,
            BSA=args.BSA,
            m=args.HA,
            k=args.WA,
            BSB=args.BSB,
            n=args.WB,
            acc_dtype=args.acc_dtype, 
            out_dtype=args.out_dtype,
            provider="torch", 
            out_dir=out_dir,
            )
    else:
        raise Exception("Unsupported operator!")
    print(res, "TFLOPS")
