import sys
sys.path.append('../')
# from utils.tvm_workloads_fp16 import *
# from utils.tvm_config import *
from typing import List, Optional, Tuple, Union

import os
import json
import argparse
from tabulate import tabulate
from utils import cuda
from utils.util import load_function_from_module

import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"


def environment_info(args) -> str:
    return str(tabulate(
        headers=[
            'Name', 'Value'
        ],
        tabular_data=[
            ['GPU', cuda.query_device_name()],
            ['Arch', cuda.query_arch()],
            ['Compute Capacity', cuda.query_compute_capability()],
            ['Current SM Clock (MHz)', cuda.query_gpu_current_clock()],
            ['Current Memory Clock (MHz)', cuda.query_memory_current_clock()],
            # ['Warmup/Number/Repeat', '{} / {} / {}'.format(args.warmup, args.number, args.repeat)]
        ]
    ))


def parse_args():
    args = argparse.ArgumentParser(description='auto benchmark script for tensor operators.')
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument("--m", type=int, required=True)
    args.add_argument("--n", type=int, required=True)
    args.add_argument("--k", type=int, required=True)
    args.add_argument("--BSA", type=int, required=True)
    args.add_argument("--BSB", type=int, required=True)
    args.add_argument("--TransA", type=str, required=True)
    args.add_argument("--TransB", type=str, required=True)
    args.add_argument("--engine", type=str, choices=['tvm_ms', 'triton', 'cutlass', 'torch'], required=True)
    args.add_argument("--target", type=str)
    args.add_argument("--num_trials", type=int, default=1000)
    # args.add_argument("--work_dir", type=str)
    args.add_argument("--log_dir", type=str, default="./results/")


    args.add_argument("--input_dtype", type=str,
                      choices=["f16", "f32"], default="f16")
    args.add_argument("--acc_dtype", type=str,
                      choices=["f16", "f32"], default="f32")
    args.add_argument("--out_dtype", type=str,
                      choices=["f16", "f32"], default="f32")
    use_rpc = args.add_mutually_exclusive_group()
    use_rpc.add_argument("--local", action="store_false", dest="use_rpc", default=False)
    use_rpc.add_argument("--rpc", action="store_true", dest="use_rpc")
    args.add_argument("--rpc-host", type=str)
    args.add_argument("--rpc-port", type=int)
    args.add_argument("--rpc-key", type=str)
    args.add_argument("--workers", type=int)
    args.add_argument("--alloc-repeat", type=int, default=1)

    args.add_argument("--cutlass-home", type=str, default="/home/yangbai/Documents/compiler/cutlass")
    parsed = args.parse_args()
    parsed.cutlass_home = parsed.cutlass_home or os.getenv("CUTLASS_HOME")
    assert (
        parsed.cutlass_home
    ), "Please specify 'CUTLASS_HOME', by either setting the environment variable or using --cutlass-home"
    parsed.profiler = f"{parsed.cutlass_home}/build/tools/profiler/cutlass_profiler"
    parsed = args.parse_args()
    parsed.target = parsed.target or os.environ.get("TVM_TARGET")
    # parsed.work_dir = parsed.work_dir or f"logs/"
    return parsed




def bench(command_line_args: Optional[str]=None):
    args = parse_args()
    print(f"current engine is {args.engine}")

    task_name = 'batch_{}_{}_{}_{}_{}_{}_{}_{}_{}_input_{}_acc_{}_output_{}'.format(
        args.workload,
        args.BSA, 
        args.BSB,
        args.m,
        args.n,
        args.k,
        args.TransA,
        args.TransB,
        args.engine, 
        args.input_dtype, 
        args.acc_dtype, 
        args.out_dtype)
    print(task_name)
    bench_dict = {
        "tvm_ms": ("engine_tvm_ms", "bench_tvm_ms"),
        "cutlass": ("engine_cutlass", "bench_cutlass"),
        "triton": ("engine_triton", "bench_triton"),
        "torch": ("engine_torch", "bench_torch"),
    }
    bench_func_tuples = bench_dict[args.engine]
    out_dir = os.path.join(args.log_dir, cuda.query_device_name(short=True), 'workloads')
    print(out_dir)
    if args.engine in ["tvm_ms"]:
        trials = args.num_trials
        task_name += "_trials_{}".format(trials)
    out_dir = os.path.join(out_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(environment_info(args))

    bench_func = load_function_from_module(bench_func_tuples[0], bench_func_tuples[1])
    bench_func(args, out_dir)
    

if __name__ == "__main__":
    bench()
