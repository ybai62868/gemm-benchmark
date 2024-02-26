import torch
import triton


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

    if TransA == "N" and TransB == "N":
        HA = m
        WA = k
        HB = k
        WB = n
        a = torch.randn((BSA, HA, HB), device='cuda', dtype=input_dtype)
        b = torch.randn((BSB, HB, WB), device='cuda', dtype=input_dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b), quantiles=quantiles)
        perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    elif TransA == "N" and TransB == "T":
        HA = m
        WA = k
        HB = n
        WB = k
        a = torch.randn((BSA, HA, WA), device='cuda', dtype=input_dtype)
        b = torch.randn((BSB, HB, WB), device='cuda', dtype=input_dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b.transpose(2, 1)), quantiles=quantiles)
        perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    elif TransA == "T" and TransB == "N":
        HA = k
        WA = m
        HB = k
        WB = n
        a = torch.randn((BSA, HA, WA), device='cuda', dtype=input_dtype)
        b = torch.randn((BSB, HB, WB), device='cuda', dtype=input_dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a.transpose(2, 1), b), quantiles=quantiles)
        perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    else:
        HA = k
        WA = m
        HB = n
        WB = k
        a = torch.randn((BSA, HA, WA), device='cuda', dtype=input_dtype)
        b = torch.randn((BSB, HB, WB), device='cuda', dtype=input_dtype)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a.transpose(2, 1), b.transpose(2, 1)), quantiles=quantiles)
        perf = lambda ms: 2 * BSC * m * n * k * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)
    

def bench_torch(args, out_dir):
    
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
            provider="torch", 
            out_dir=out_dir,
            )
    else:
        raise Exception("Unsupported operator!")
    print(res, "TFLOPS")


