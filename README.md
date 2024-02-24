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
python bench.py --engine tvm_ms --workload GEMM --BSA 10 --HA 512 --WA 1024 --BSB 10 --WB 1024 --HB 512
```

### Results
```shell
current engine is tvm_ms
batch_GEMM_10_10_512_1024_512_1024_tvm_ms_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
2024-02-24 16:26:06.024 INFO LocalRunner: max_workers = 1
[10, 512, 512, 1024]
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
2024-02-24 16:26:06.326 INFO LocalBuilder: max_workers = 24
hhhhh
2024-02-24 16:26:06.575 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_1024_512_1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-24 16:26:06.576 INFO Logging directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_1024_512_1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/logs
2024-02-24 16:26:06.576 INFO Working directory: ./results/RTX3090/workloads/batch_GEMM_10_10_512_1024_512_1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000
2024-02-24 16:26:06.576 INFO Creating JSONDatabase. Workload at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_1024_512_1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_workload.json. Tuning records at: ./results/RTX3090/workloads/batch_GEMM_10_10_512_1024_512_1024_tvm_ms_input_f16_acc_f32_output_f32_trials_1000/database_tuning_record.json
2024-02-24 16:26:07.025 INFO Initializing Task #0: "main"
2024-02-24 16:26:07.053 INFO 
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |            N/A |          N/A |                   N/A |      0 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 0
Total latency (us): 0

2024-02-24 16:26:07.053 INFO Scheduler picks Task #0: "main"
2024-02-24 16:26:20.899 INFO Sending 64 sample(s) to builder
2024-02-24 16:26:29.649 INFO Sending 64 sample(s) to runner
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
  from pandas import MultiIndex, Int64Index
/home/yangbai/anaconda3/lib/python3.10/site-packages/xgboost/training.py:17: UserWarning: Old style callback is deprecated.  See: https://xgboost.readthedocs.io/en/latest/python/callbacks.html
  warnings.warn(f'Old style callback is deprecated.  See: {link}', UserWarning)
2024-02-24 16:26:50.223 INFO [Updated] Task #0: "main"
 ID | Name |       FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Terminated 
---------------------------------------------------------------------------------------------------------------
  0 | main | 5368709120 |      1 |     23404.8224 |     229.3847 |              229.3847 |     64 |            
---------------------------------------------------------------------------------------------------------------
Total trials: 64
Total latency (us): 229.385
```


### Run the benchmark with PyTorch
```shell
python bench.py --engine torch --workload GEMM --BSA 10 --HA 512 --WA 1024 --BSB 10 --WB 1024 --HB 512
```

```shell
current engine is torch
batch_GEMM_10_10_512_1024_512_1024_torch_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
62.045914853843605 TFLOPS

```

### Run the benchmark with Triton
```shell
python bench.py --engine triton --workload GEMM --BSA 10 --HA 512 --WA 1024 --BSB 10 --WB 1024 --HB 512
```
```shell
current engine is triton
batch_GEMM_10_10_512_1024_512_1024_triton_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yangbai/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
triton_output=tensor([[[-3.9771e-01, -1.0359e+01, -4.7930e+00,  ..., -1.3891e+01,
           2.2781e+01, -5.7438e+01],
         [ 1.2344e+01, -5.3156e+01, -3.2531e+01,  ...,  1.0547e+01,
           4.9023e+00,  3.1438e+01],
         [ 5.5234e+00,  2.3359e+00, -6.9766e+00,  ..., -1.4883e+01,
          -3.5625e+01, -1.5957e+00],
         ...,
         [ 1.2773e+00,  6.0508e+00,  7.2125e+01,  ...,  1.1461e+01,
          -3.2125e+01, -1.6688e+01],
         [-1.5438e+01,  1.1594e+01,  1.1656e+01,  ...,  2.6312e+01,
           2.8438e+00, -5.5312e+01],
         [-2.5078e+01, -9.5391e+00, -1.0117e+01,  ..., -4.7656e+01,
           7.8203e+00,  5.4883e+00]],

        [[ 2.2875e+01,  4.1469e+01,  1.6219e+01,  ...,  4.2531e+01,
           1.1281e+01, -9.7891e+00],
         [-5.4094e+01,  5.1758e+00,  9.5508e-01,  ..., -2.4391e+01,
           1.1266e+01,  1.3945e+01],
         [-1.4050e-01, -2.3844e+01,  2.1844e+01,  ..., -1.1617e+01,
           2.6938e+01, -3.4500e+01],
         ...,
         [ 3.7781e+01, -7.6250e+00, -2.9188e+01,  ...,  8.3750e+00,
          -1.0062e+01,  5.4375e+01],
         [-2.2719e+01,  2.5172e+01,  6.7334e-01,  ...,  3.5125e+01,
           1.1148e+01,  4.3906e+01],
         [ 2.2984e+01,  4.0741e-02,  5.5039e+00,  ..., -3.2781e+01,
           1.3914e+01,  4.8094e+01]],

        [[-5.7734e+00,  4.2125e+01, -2.6906e+01,  ...,  3.8594e+00,
           1.8953e+01, -1.0633e+01],
         [-2.5672e+01,  1.9156e+01,  1.0609e+01,  ...,  7.0117e+00,
          -2.3711e+00,  5.0781e+00],
         [ 1.5023e+01,  9.3281e+00,  5.3812e+01,  ...,  1.4898e+01,
           9.3359e+00, -2.3953e+01],
         ...,
         [ 5.6211e+00,  2.6828e+01,  2.7453e+01,  ..., -1.8078e+01,
           3.7754e+00,  5.6312e+01],
         [-2.6781e+01,  2.5488e+00,  8.0625e+01,  ...,  7.2875e+01,
           1.9629e+00,  3.4062e+01],
         [ 1.8812e+01,  3.7969e+01,  1.0273e+01,  ...,  1.6766e+01,
          -4.3406e+01,  3.4023e+00]],

        ...,

        [[ 5.3406e+01, -5.8320e+00,  2.8234e+01,  ...,  2.4938e+01,
           2.4438e+01,  2.9156e+01],
         [ 3.0516e+01, -2.1266e+01, -3.3062e+01,  ...,  1.3609e+01,
           1.2766e+01,  5.3031e+01],
         [-4.4570e+00, -6.0438e+01,  3.8156e+01,  ..., -3.9531e+01,
           1.3875e+01,  5.0742e+00],
         ...,
         [-1.9250e+01, -2.2754e+00,  1.0984e+01,  ...,  1.6391e+01,
           8.0762e-01, -1.1025e+00],
         [ 4.3719e+01,  7.4258e+00, -1.6672e+01,  ...,  2.9438e+01,
          -2.8406e+01, -4.6469e+01],
         [ 2.7062e+01,  4.2305e+00,  4.1797e+00,  ..., -4.9570e+00,
           3.0609e+01, -1.4312e+01]],

        [[-2.4906e+01,  1.4375e+01,  1.3891e+01,  ..., -2.5781e+01,
          -3.3375e+01,  1.3859e+01],
         [ 1.6660e+00, -2.9062e+01, -2.8047e+01,  ..., -5.0234e+00,
          -4.1562e+01,  1.5820e+01],
         [ 6.4336e+00, -2.1055e+00, -2.1875e+01,  ...,  6.4531e+00,
           4.9156e+01, -1.4656e+01],
         ...,
         [ 9.0625e+00, -2.8562e+01,  3.6688e+01,  ..., -3.4938e+01,
           5.5281e+01, -6.1680e+00],
         [ 2.5094e+01, -3.1938e+01, -1.5836e+01,  ..., -1.1508e+01,
           5.8320e+00,  2.3828e+01],
         [-1.0039e+01, -2.1766e+01,  4.4844e+01,  ..., -5.6312e+01,
           2.8297e+01,  4.3125e+01]],

        [[-4.1188e+01, -6.8320e+00, -1.9000e+01,  ..., -2.3062e+01,
           4.7062e+01, -2.9906e+01],
         [ 1.6953e+01, -1.6859e+01, -2.6562e+01,  ...,  5.0750e+01,
          -5.2891e+00,  1.1412e+02],
         [-7.9125e+01,  2.2031e+01, -5.6938e+01,  ...,  3.3418e+00,
          -1.6734e+01,  5.1367e+00],
         ...,
         [-5.4312e+01,  4.3438e+00, -4.6562e+01,  ..., -5.2031e+00,
          -7.9102e+00,  1.0586e+01],
         [-4.9031e+01,  2.4547e+01, -1.2461e+01,  ..., -2.8297e+01,
           3.5531e+01, -6.2844e+01],
         [-1.1492e+01,  3.4121e+00, -4.4438e+01,  ..., -1.3383e+01,
           9.7500e+00, -3.3812e+01]]], device='cuda:0', dtype=torch.float16)
torch_output=tensor([[[-3.9771e-01, -1.0359e+01, -4.7930e+00,  ..., -1.3891e+01,
           2.2781e+01, -5.7438e+01],
         [ 1.2344e+01, -5.3156e+01, -3.2531e+01,  ...,  1.0547e+01,
           4.9023e+00,  3.1438e+01],
         [ 5.5234e+00,  2.3359e+00, -6.9766e+00,  ..., -1.4883e+01,
          -3.5625e+01, -1.5957e+00],
         ...,
         [ 1.2773e+00,  6.0508e+00,  7.2125e+01,  ...,  1.1461e+01,
          -3.2125e+01, -1.6688e+01],
         [-1.5438e+01,  1.1594e+01,  1.1656e+01,  ...,  2.6312e+01,
           2.8438e+00, -5.5312e+01],
         [-2.5078e+01, -9.5391e+00, -1.0117e+01,  ..., -4.7656e+01,
           7.8203e+00,  5.4883e+00]],

        [[ 2.2875e+01,  4.1469e+01,  1.6219e+01,  ...,  4.2531e+01,
           1.1281e+01, -9.7891e+00],
         [-5.4094e+01,  5.1758e+00,  9.5508e-01,  ..., -2.4391e+01,
           1.1266e+01,  1.3945e+01],
         [-1.4050e-01, -2.3844e+01,  2.1844e+01,  ..., -1.1617e+01,
           2.6938e+01, -3.4500e+01],
         ...,
         [ 3.7781e+01, -7.6250e+00, -2.9188e+01,  ...,  8.3750e+00,
          -1.0062e+01,  5.4375e+01],
         [-2.2719e+01,  2.5172e+01,  6.7334e-01,  ...,  3.5125e+01,
           1.1148e+01,  4.3906e+01],
         [ 2.2984e+01,  4.0710e-02,  5.5039e+00,  ..., -3.2781e+01,
           1.3914e+01,  4.8094e+01]],

        [[-5.7734e+00,  4.2125e+01, -2.6906e+01,  ...,  3.8594e+00,
           1.8953e+01, -1.0633e+01],
         [-2.5672e+01,  1.9156e+01,  1.0609e+01,  ...,  7.0117e+00,
          -2.3711e+00,  5.0781e+00],
         [ 1.5023e+01,  9.3281e+00,  5.3812e+01,  ...,  1.4898e+01,
           9.3359e+00, -2.3953e+01],
         ...,
         [ 5.6211e+00,  2.6828e+01,  2.7453e+01,  ..., -1.8078e+01,
           3.7754e+00,  5.6312e+01],
         [-2.6781e+01,  2.5488e+00,  8.0625e+01,  ...,  7.2875e+01,
           1.9629e+00,  3.4062e+01],
         [ 1.8812e+01,  3.7969e+01,  1.0273e+01,  ...,  1.6766e+01,
          -4.3406e+01,  3.4023e+00]],

        ...,

        [[ 5.3406e+01, -5.8320e+00,  2.8234e+01,  ...,  2.4938e+01,
           2.4438e+01,  2.9156e+01],
         [ 3.0516e+01, -2.1266e+01, -3.3062e+01,  ...,  1.3609e+01,
           1.2766e+01,  5.3031e+01],
         [-4.4570e+00, -6.0438e+01,  3.8156e+01,  ..., -3.9531e+01,
           1.3875e+01,  5.0742e+00],
         ...,
         [-1.9250e+01, -2.2754e+00,  1.0984e+01,  ...,  1.6391e+01,
           8.0762e-01, -1.1025e+00],
         [ 4.3719e+01,  7.4258e+00, -1.6672e+01,  ...,  2.9438e+01,
          -2.8406e+01, -4.6469e+01],
         [ 2.7062e+01,  4.2305e+00,  4.1797e+00,  ..., -4.9570e+00,
           3.0609e+01, -1.4312e+01]],

        [[-2.4906e+01,  1.4375e+01,  1.3891e+01,  ..., -2.5781e+01,
          -3.3375e+01,  1.3859e+01],
         [ 1.6660e+00, -2.9062e+01, -2.8047e+01,  ..., -5.0234e+00,
          -4.1562e+01,  1.5820e+01],
         [ 6.4336e+00, -2.1055e+00, -2.1875e+01,  ...,  6.4531e+00,
           4.9156e+01, -1.4656e+01],
         ...,
         [ 9.0625e+00, -2.8562e+01,  3.6688e+01,  ..., -3.4938e+01,
           5.5281e+01, -6.1680e+00],
         [ 2.5094e+01, -3.1938e+01, -1.5836e+01,  ..., -1.1508e+01,
           5.8320e+00,  2.3828e+01],
         [-1.0039e+01, -2.1766e+01,  4.4844e+01,  ..., -5.6312e+01,
           2.8297e+01,  4.3125e+01]],

        [[-4.1188e+01, -6.8320e+00, -1.9000e+01,  ..., -2.3062e+01,
           4.7062e+01, -2.9906e+01],
         [ 1.6953e+01, -1.6859e+01, -2.6562e+01,  ...,  5.0750e+01,
          -5.2891e+00,  1.1412e+02],
         [-7.9125e+01,  2.2031e+01, -5.6938e+01,  ...,  3.3418e+00,
          -1.6734e+01,  5.1367e+00],
         ...,
         [-5.4312e+01,  4.3438e+00, -4.6562e+01,  ..., -5.2031e+00,
          -7.9102e+00,  1.0586e+01],
         [-4.9031e+01,  2.4547e+01, -1.2461e+01,  ..., -2.8297e+01,
           3.5531e+01, -6.2844e+01],
         [-1.1492e+01,  3.4121e+00, -4.4438e+01,  ..., -1.3383e+01,
           9.7500e+00, -3.3812e+01]]], device='cuda:0', dtype=torch.float16)
✅ Triton and Torch match
70.84972664319672 TFLOPS
```

### Run the benchmark with CUTLASS
```shell
python bench.py --engine cutlass --workload GEMM --BSA 10 --HA 512 --WA 1024 --BSB 10 --WB 1024 --HB 512
```
```shell
batch_GEMM_10_10_512_1024_512_1024_cutlass_input_f16_acc_f32_output_f32
./results/RTX3090/workloads
/home/yang/Desktop/github_yang/gemm-benchmark/utils/deviceQuery/deviceQuery
Running: GEMM-10-f32-f32
GEMM-10-f32-f32: 56.8623046875 TFLOPS
Full benchmark results have been written to ./results/RTX3090/workloads/batch_GEMM_10_10_512_1024_512_1024_cutlass_input_f16_acc_f32_output_f32/GEMM-10-f32-f32.csv
```
