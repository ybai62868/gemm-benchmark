import sys
sys.path.append('../')
from typing import Optional
import tvm
from tvm import meta_schedule as ms
from utils.tvm_workloads_fp16 import create_te_workload_f16
from utils.tvm_config import sch_rules_tensor_core, postprocs_tensor_core, get_search_config
from tvm.target import Target


__all__ = ['bench_tvm_ansor', 'bench_tvm_ms']

def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)


def prepare_runner(args):
    if args.use_rpc:
        rpc_host = args.rpc_host or os.environ.get("TVM_RPC_HOST")
        rpc_port = args.rpc_port or int(os.environ.get("TVM_RPC_PORT"))
        rpc_key = args.rpc_key or os.environ.get("TVM_RPC_KEY")
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc_host,
            tracker_port=rpc_port,
            tracker_key=rpc_key,
            session_timeout_sec=60,
        )
        workers = args.workers or rpc_config.count_num_servers(allow_missing=False)
        args.runner = partial(
            ms.runner.RPCRunner, rpc_config=rpc_config, max_workers=workers
        )
    else:
        args.runner = ms.runner.LocalRunner
    args.runner = args.runner(
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )
    )

def bench_tvm_ansor(args, out_dir):
    prepare_runner(args)
    if args.out_dtype == "f16":
        args.out_dtype = "float16"
    elif args.out_dtype == "f32":
        args.out_dtype = "float32"
    else:
        raise Exception("Unsupported dtype")
    mod = create_te_workload_f16(
        args.workload, batch_size=args.batch_size, out_dtype=args.out_dtype
    )
    print("start tuning with meta schedule ...")
    sch = ms.tune_tir(
        mod=mod,
        target=Target(args.target),
        config=get_search_config(args.num_trials, args.num_trials),
        work_dir=out_dir,
        runner=args.runner,
    )

    if sch is None:
        print("No valid schedule found!")
        exit()
    print(sch.mod.script())
    print(sch.trace)


def bench_tvm_ms(args, out_dir):
    prepare_runner(args)
    if args.out_dtype == "f16":
        args.out_dtype = "float16"
        from tvm.meta_schedule.testing import tir_tensor_intrin_fp16
    elif args.out_dtype == "f32":
        args.out_dtype = "float32"
        from tvm.meta_schedule.testing import tir_tensor_intrin
    else:
        raise Exception("Unsupported dtype")
    mod = create_te_workload_f16(
        args.workload, 
        args.BSA,
        args.BSB,
        args.HA,
        args.WB,
        args.WA,
        out_dtype=args.out_dtype
    )
    print(mod)
    print("start tuning with meta schedule ...")
    sch = ms.tune_tir(
        mod=mod,
        target=Target(args.target),
        config=get_search_config(args.num_trials, args.num_trials),
        work_dir=out_dir,
        builder=ms.builder.LocalBuilder(f_build=cuda_build),
        runner=args.runner,
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
    )

    if sch is None:
        print("No valid schedule found!")
        exit()
    print(sch.mod.script())
    print(sch.trace)    

