# gemm-benchmark on NV GPUs
Python based gemm benchmark for tensor computation
```shell
1. cd utils/deviceQuery
2. make -j8
```

### LLVM Installation for TVM Meta-Schedule
```shell
1. git clone https://github.com/llvm/llvm-project.git
2. cd llvm-project
3. git checkout llvmorg-13.0.1
4. mkdir build && cd build
5. cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="mlir;clang" \
    -DLLVM_TARGETS_TO_BUILD="host;RISCV;NVPTX" \
    -DMLIR_ENABLE_CUDA_RUNNER=ON \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
 6. ninja -j12
 7. ninja check-all
 8. sudo ninja install 
 9. export PATH=/home/yang/Desktop/asplos-tvm/llvm-project/build/bin:$PATH
```

### TVM Installation with Meta-Schedule
```shell
1. git clone --recursive https://github.com/Hzfengsy/asplos-tvm.git
2. cd asplos-tvm
3. mkdir build && cd build
4. cp ../cmake/config.cmake ./
5.  set(LLVM ON) set(CUDA ON)
6. cmake .. && make -j12
7. pip install synr xgboost==1.5
8. export TVM_HOME=/path/to/tvm
9. export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
10. export TVM_TARGET="nvidia/geforce-rtx-3090"
```

### CUTLASS Installation
```shell
1. git clone https://github.com/NVIDIA/cutlass.git 
2. git checkout v2.9.1
3. export CUDACXX=/usr/local/cuda/bin/nvcc
4. mkdir build && cd build
5. cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_KERNELS=all
6. make cutlass_profiler -j16

```


### Run the benchmark with TVM Meta-Schedule
```shell
# python bench.py --engine tvm_ms --workload GEMM-1024-1024-1024
python bench.py --engine tvm_ms --workload GEMM --BSA 10 --HA 512 --WA 1024 --BSB 10 --WB 1024 --HB 512
```

### Results
```shell
current engine is tvm_ms
batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f32_output_f32
2024-02-22 16:29:51.591 INFO LocalRunner: max_workers = 1
primfn(var_X: handle, var_Y: handle, var_Z: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_1: Pointer(global float16), float16, [1, 1024, 1024], []),
             Y: Buffer(Y_1: Pointer(global float16), float16, [1, 1024, 1024], []),
             Z: Buffer(Z_1: Pointer(global float32), float32, [1, 1024, 1024], [])}
  buffer_map = {var_X: X, var_Y: Y, var_Z: Z} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int32, 0, 1) {
      for (i1: int32, 0, 1024) {
        for (i2: int32, 0, 1024) {
          for (i3: int32, 0, 1024) {
            block([1, 1024, 1024, tir.reduce_axis(0, 1024)], "Z") as [b, i, j, k] {
              bind(b, i0)
              bind(i, i1)
              bind(j, i2)
              bind(k, i3)
              tir.reads([X[0, i, k], Y[0, k, j]])
              tir.writes([Z[b, i, j]])
              with init() {
                Z[b, i, j] = 0f32
              }
              Z[b, i, j] = (Z[b, i, j] + (cast(float32, X[0, i, k])*cast(float32, Y[0, k, j])))
          }
        }
      }
    }
}

start tuning with meta schedule ...
2024-02-22 16:29:51.896 INFO LocalBuilder: max_workers = 24
hhhhh
2024-02-22 16:29:52.152 INFO Logging directory: ./results/RTX3090/workloads/batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-22 16:29:52.152 INFO Logging directory: ./results/RTX3090/workloads/batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-22 16:29:52.152 INFO Working directory: ./results/RTX3090/workloads/batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000
2024-02-22 16:29:52.152 INFO Creating JSONDatabase. Workload at: ./results/RTX3090/workloads/batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_workload.json. Tuning records at: ./results/RTX3090/workloads/batch_size_1_GEMM-1024-1024-1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_tuning_record.json
2024-02-22 16:29:52.153 INFO Initializing Task #0: "main"
2024-02-22 16:29:52.180 INFO 
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 2147483648 |      1 |            N/A |          N/A |                   N/A |      0 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

2024-02-22 16:29:52.180 INFO Scheduler picks Task #0: "main"
2024-02-22 16:30:03.759 INFO Sending 64 sample(s) to builder
2024-02-22 16:30:08.379 INFO Sending 64 sample(s) to runner
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
2024-02-22 16:30:18.430 INFO [Updated] Task #0: "main"
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 2147483648 |      1 |     44580.0983 |      48.1714 |               48.1714 |     64 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 64

```


### Run the benchmark with PyTorch
```shell
python bench.py --engine torch --workload GEMM-1024-1024-1024
```

```shell
current engine is torch
batch_size_1_GEMM-1024-1024-1024_torch_input_f16_acc_f32_output_f32
45.59025985792771 ms
```

### Run the benchmark with Tritopn
```shell
python bench.py --engine triton --workload GEMM-1024-1024-1024
```
```shell
current engine is triton
batch_size_1_GEMM-1024-1024-1024_triton_input_f16_acc_f32_output_f32
53.7731300651556 ms
```

### Run the benchmark with CUTLASS
```shell
python bench.py --engine cutlass --workload GEMM-1024-1024-1024
```
```shell
current engine is cutlass
batch_size_1_GEMM-1024-1024-1024_cutlass_input_f16_acc_f32_output_f32
Running: GEMM-1024-1024-1024-1-f32-f32
GEMM-1024-1024-1024-1-f32-f32: 24.5298828125 TFLOPS
Full benchmark results have been written to ./results/RTX3090/workloads/batch_size_1_GEMM-1024-1024-1024_cutlass_input_f16_acc_f32_output_f32/GEMM-1024-1024-1024-1-f32-f32.csv
```
