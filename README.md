# gemm-benchmark
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
python bench.py --engine tvm_ms --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA N --TransB T
current engine is tvm_ms
batch_GEMM_10_10_512_512_1024_N_T_tvm_ms_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
2024-02-26 17:49:26.865 INFO LocalRunner: max_workers = 1
primfn(var_X: handle, var_Y: handle, var_Z: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_1: Pointer(global float16), float16, [10, 512, 1024], []),
             Y: Buffer(Y_1: Pointer(global float16), float16, [10, 512, 1024], []),
             Z: Buffer(Z_1: Pointer(global float32), float32, [10, 512, 512], [])}
  buffer_map = {var_X: X, var_Y: Y, var_Z: Z} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int32, 0, 10) {
      for (i1: int32, 0, 512) {
        for (i2: int32, 0, 512) {
          for (i3: int32, 0, 1024) {
            block([10, 512, 512, tir.reduce_axis(0, 1024)], "Z") as [b, i, j, k] {
              bind(b, i0)
              bind(i, i1)
              bind(j, i2)
              bind(k, i3)
              tir.reads([X[b, i, k], Y[b, j, k]])
              tir.writes([Z[b, i, j]])
              with init() {
                Z[b, i, j] = 0f32
              }
              Z[b, i, j] = (Z[b, i, j] + (cast(float32, X[b, i, k])*cast(float32, Y[b, j, k])))
          }
        }
      }
    }
}

start tuning with meta schedule ...
2024-02-26 17:49:27.166 INFO LocalBuilder: max_workers = 24
hhhhh
2024-02-26 17:49:27.422 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:49:27.422 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:49:27.422 INFO Working directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000
2024-02-26 17:49:27.422 INFO Creating JSONDatabase. Workload at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_workload.json. Tuning records at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_tuning_record.json
2024-02-26 17:49:27.505 INFO Initializing Task #0: "main"
2024-02-26 17:49:27.536 INFO 
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |            N/A |          N/A |                   N/A |      0 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

2024-02-26 17:49:27.536 INFO Scheduler picks Task #0: "main"
2024-02-26 17:49:41.048 INFO Sending 64 sample(s) to builder
2024-02-26 17:49:48.840 INFO Sending 64 sample(s) to runner
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
2024-02-26 17:50:13.101 INFO [Updated] Task #0: "main"
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |     57412.0973 |      93.5118 |               93.5118 |     64 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 93.5118
```

```shell
python bench.py --engine tvm_ms --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA T --TransB N
current engine is tvm_ms
batch_GEMM_10_10_512_512_1024_T_N_tvm_ms_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
2024-02-26 17:51:08.859 INFO LocalRunner: max_workers = 1
primfn(var_X: handle, var_Y: handle, var_Z: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_1: Pointer(global float16), float16, [10, 1024, 512], []),
             Y: Buffer(Y_1: Pointer(global float16), float16, [10, 1024, 512], []),
             Z: Buffer(Z_1: Pointer(global float32), float32, [10, 512, 512], [])}
  buffer_map = {var_X: X, var_Y: Y, var_Z: Z} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int32, 0, 10) {
      for (i1: int32, 0, 512) {
        for (i2: int32, 0, 512) {
          for (i3: int32, 0, 1024) {
            block([10, 512, 512, tir.reduce_axis(0, 1024)], "Z") as [b, i, j, k] {
              bind(b, i0)
              bind(i, i1)
              bind(j, i2)
              bind(k, i3)
              tir.reads([X[b, k, i], Y[b, k, j]])
              tir.writes([Z[b, i, j]])
              with init() {
                Z[b, i, j] = 0f32
              }
              Z[b, i, j] = (Z[b, i, j] + (cast(float32, X[b, k, i])*cast(float32, Y[b, k, j])))
          }
        }
      }
    }
}

start tuning with meta schedule ...
2024-02-26 17:51:09.169 INFO LocalBuilder: max_workers = 24
hhhhh
2024-02-26 17:51:09.420 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:51:09.421 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:51:09.421 INFO Working directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000
2024-02-26 17:51:09.421 INFO Creating JSONDatabase. Workload at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_workload.json. Tuning records at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_tuning_record.json
2024-02-26 17:51:09.422 INFO Initializing Task #0: "main"
2024-02-26 17:51:09.450 INFO 
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |            N/A |          N/A |                   N/A |      0 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

2024-02-26 17:51:09.450 INFO Scheduler picks Task #0: "main"
2024-02-26 17:51:23.017 INFO Sending 64 sample(s) to builder
2024-02-26 17:51:39.480 INFO Sending 64 sample(s) to runner
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
2024-02-26 17:51:52.878 INFO [Updated] Task #0: "main"
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |     26579.7810 |     201.9847 |              201.9847 |     64 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 201.985

```

```shell
python bench.py --engine tvm_ms --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA T --TransB T
current engine is tvm_ms
batch_GEMM_10_10_512_512_1024_T_T_tvm_ms_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
2024-02-26 17:52:14.148 INFO LocalRunner: max_workers = 1
primfn(var_X: handle, var_Y: handle, var_Z: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_1: Pointer(global float16), float16, [10, 1024, 512], []),
             Y: Buffer(Y_1: Pointer(global float16), float16, [10, 512, 1024], []),
             Z: Buffer(Z_1: Pointer(global float32), float32, [10, 512, 512], [])}
  buffer_map = {var_X: X, var_Y: Y, var_Z: Z} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int32, 0, 10) {
      for (i1: int32, 0, 512) {
        for (i2: int32, 0, 512) {
          for (i3: int32, 0, 1024) {
            block([10, 512, 512, tir.reduce_axis(0, 1024)], "Z") as [b, i, j, k] {
              bind(b, i0)
              bind(i, i1)
              bind(j, i2)
              bind(k, i3)
              tir.reads([X[b, k, i], Y[b, j, k]])
              tir.writes([Z[b, i, j]])
              with init() {
                Z[b, i, j] = 0f32
              }
              Z[b, i, j] = (Z[b, i, j] + (cast(float32, X[b, k, i])*cast(float32, Y[b, j, k])))
          }
        }
      }
    }
}

start tuning with meta schedule ...
2024-02-26 17:52:14.454 INFO LocalBuilder: max_workers = 24
hhhhh
2024-02-26 17:52:14.718 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:52:14.718 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:52:14.718 INFO Working directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000
2024-02-26 17:52:14.718 INFO Creating JSONDatabase. Workload at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_workload.json. Tuning records at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_T_T_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_tuning_record.json
2024-02-26 17:52:14.763 INFO Initializing Task #0: "main"
2024-02-26 17:52:14.791 INFO 
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |            N/A |          N/A |                   N/A |      0 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

2024-02-26 17:52:14.791 INFO Scheduler picks Task #0: "main"
2024-02-26 17:52:28.603 INFO Sending 64 sample(s) to builder
2024-02-26 17:52:49.406 INFO Sending 64 sample(s) to runner
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
2024-02-26 17:53:07.494 INFO [Updated] Task #0: "main"
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |     26706.9909 |     201.0226 |              201.0226 |     64 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 201.023
```

```shell
python bench.py --engine tvm_ms --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA N --TransB N
current engine is tvm_ms
batch_GEMM_10_10_512_512_1024_N_N_tvm_ms_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
2024-02-26 17:55:30.266 INFO LocalRunner: max_workers = 1
primfn(var_X: handle, var_Y: handle, var_Z: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {X: Buffer(X_1: Pointer(global float16), float16, [10, 512, 1024], []),
             Y: Buffer(Y_1: Pointer(global float16), float16, [10, 1024, 512], []),
             Z: Buffer(Z_1: Pointer(global float32), float32, [10, 512, 512], [])}
  buffer_map = {var_X: X, var_Y: Y, var_Z: Z} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int32, 0, 10) {
      for (i1: int32, 0, 512) {
        for (i2: int32, 0, 512) {
          for (i3: int32, 0, 1024) {
            block([10, 512, 512, tir.reduce_axis(0, 1024)], "Z") as [b, i, j, k] {
              bind(b, i0)
              bind(i, i1)
              bind(j, i2)
              bind(k, i3)
              tir.reads([X[b, i, k], Y[b, k, j]])
              tir.writes([Z[b, i, j]])
              with init() {
                Z[b, i, j] = 0f32
              }
              Z[b, i, j] = (Z[b, i, j] + (cast(float32, X[b, i, k])*cast(float32, Y[b, k, j])))
          }
        }
      }
    }
}

start tuning with meta schedule ...
2024-02-26 17:55:30.579 INFO LocalBuilder: max_workers = 24
hhhhh
2024-02-26 17:55:30.837 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:55:30.838 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-26 17:55:30.838 INFO Working directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000
2024-02-26 17:55:30.838 INFO Creating JSONDatabase. Workload at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_workload.json. Tuning records at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_N_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_tuning_record.json
2024-02-26 17:55:30.838 INFO Initializing Task #0: "main"
2024-02-26 17:55:30.868 INFO 
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |            N/A |          N/A |                   N/A |      0 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

2024-02-26 17:55:30.868 INFO Scheduler picks Task #0: "main"
2024-02-26 17:55:44.699 INFO Sending 64 sample(s) to builder
2024-02-26 17:55:52.089 INFO Sending 64 sample(s) to runner
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
2024-02-26 17:56:10.128 INFO [Updated] Task #0: "main"
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |     30468.7195 |     176.2040 |              176.2040 |     64 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 176.204
```






### Run the benchmark with PyTorch
```shell
python bench.py --engine torch --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA T --TransB N
current engine is torch
batch_GEMM_10_10_512_512_1024_T_N_torch_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
58.25422133909627 TFLOPS
```


### Run the benchmark with Triton
```shell
python bench.py --engine triton --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA N --TransB N
```
```shell
current engine is triton
batch_GEMM_10_10_512_512_1024_N_N_triton_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
✅ Triton and Torch match
70.84972664319672 TFLOPS
```

### Run the benchmark with CUTLASS
```shell
python bench.py --engine cutlass --workload GEMM --BSA 10 --BSB 10 --m 512 --k 1024 --n 512 --TransA N --TransB N
```
```shell
current engine is cutlass
batch_GEMM_10_10_512_512_1024_N_N_cutlass_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
Running: GEMM-10-f32-f32
GEMM-10-f32-f32: 56.090625 TFLOPS
Full benchmark results have been written to ./results/RTX3090/workloads/batch_GEMM_10_10_512_512_1024_N_N_cutlass_input_f16_acc_f32_output_f32/GEMM-10-f32-f32.csv
```
