# GPU Architecture Exploration and Performance Optimization

**From GPU Architecture Fundamentals to GPU-Aware Deep Learning Training**

An experimental study of NVIDIA GPU architecture, CUDA execution, memory hierarchy, parallel computing, Tensor Cores, and deep learning performance optimization using PyTorch and Triton.

## Project Overview

GPUs are designed for massively parallel workloads, but achieving high performance requires more than simply moving computation onto a GPU.

This project explores how GPU hardware works and how architectural knowledge can be translated into practical performance improvements.

Across 14 experiments, I investigated GPU execution, memory transfers, thread scheduling, memory access patterns, matrix multiplication, compute throughput, and mixed-precision training.

The project concludes with a comparison between conventional ResNet-50 training and a GPU-aware optimized training approach on an NVIDIA Tesla T4.

## Key Result

The final experiment demonstrated a **3.76× training speedup** using a combination of GPU-aware optimizations.

| Metric | Conventional | GPU-aware |
|---|---:|---:|
| Training time | 241.80 sec | 64.31 sec |
| Throughput | 82.71 images/sec | 311.01 images/sec |
| Peak GPU memory | 6.11 GB | 3.57 GB |
| Validation accuracy | 34.75% | 38.15% |

**Performance improvements:**
- 3.76× faster training
- 73.4% reduction in training time
- 41.6% reduction in peak GPU memory
- 3.40 percentage-point improvement in validation accuracy

These results are from one benchmark run under the experimental conditions described below.

## Hardware and Software Environment

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla T4 × 2 |
| CUDA Cores | 2,560 per GPU |
| Tensor Cores | 320 per GPU |
| Streaming Multiprocessors | 40 per GPU |
| Compute Capability | 7.5 |
| GPU Memory | Approximately 14.56 GiB per GPU |
| Warp Size | 32 threads |
| GPU Interconnect | PCIe / PHB topology |

The two GPUs were independent devices; their memory was not pooled.

### Software

- Python 3.13.16
- PyTorch 2.11.0+cu130
- CUDA 13.0
- Triton
- NumPy
- Kaggle Notebook Environment

## Experiments

The project contains 14 experiments organized around understanding GPU architecture and applying it to real workloads.

### Phase 1 — GPU Architecture Fundamentals

| No. | Experiment | Key Observation |
|---|---|---|
| 01 | GPU Hardware Discovery | Identified GPU specifications, SM count, warp size and memory |
| 02 | GPU Topology | Investigated GPU-to-GPU connectivity and PCIe topology |
| 03 | CPU ↔ GPU Memory Transfer | Pinned memory achieved approximately 2.5× higher effective transfer bandwidth |
| 04 | CUDA Block-to-SM Mapping | Observed how CUDA blocks were distributed across SMs |

### Phase 2 — GPU Execution and Memory Optimization

| No. | Experiment | Key Observation |
|---|---|---|
| 05 | Warp Configuration and Registers | 8 warps performed best among tested configurations |
| 06 | GPU Memory Hierarchy and Cache Policies | Compared `.ca` and `.cg` load policies |
| 07 | Memory Coalescing | Coalesced access was approximately 4.5× faster than strided access |
| 08 | Tiled Matrix Multiplication | Best tested configuration achieved 2.201 TFLOPS |

### Phase 3 — GPU Compute and Acceleration

| No. | Experiment | Key Observation |
|---|---|---|
| 09 | CUDA Arithmetic Throughput | Observed increased throughput with greater arithmetic intensity |
| 10 | Tensor Cores and Precision | FP16 achieved approximately 12.4× the FP32 throughput in the benchmark |
| 11 | Block Size and Warp Configuration | Best tested configuration was 512 elements per block with 8 warps |

### Phase 4 — Deep Learning and Performance Engineering

| No. | Experiment | Key Observation |
|---|---|---|
| 12 | ResNet-50 Training | Implemented actual deep learning training on the Tesla T4 |
| 13 | GPU Memory Profiling | Investigated allocated, reserved and peak GPU memory |
| 14 | Conventional vs GPU-aware Training | Achieved 3.76× training speedup |

## Important Experimental Results

### 1. Pinned Memory vs Pageable Memory

Pinned memory improved effective host-to-device and device-to-host transfer bandwidth.

For the 256 MiB transfer:

| Memory Type | H2D Bandwidth |
|---|---:|
| Pageable | 4.396 GiB/s |
| Pinned | 11.497 GiB/s |

This demonstrates the importance of efficient CPU–GPU data transfers.

### 2. Memory Coalescing

| Access Pattern | Execution Time |
|---|---:|
| Coalesced | 0.1457 ms |
| Strided-8 | 0.6559 ms |

Coalesced access was approximately 4.5× faster.

### 3. Tiled Matrix Multiplication

Best tested configuration:

| Parameter | Value |
|---|---:|
| Matrix dimensions | 1024 × 1024 |
| BM | 32 |
| BN | 64 |
| BK | 32 |
| Warps | 4 |
| Execution time | 0.9755 ms |
| Performance | 2.201 TFLOPS |

Increasing the tile depth to 64 increased shared memory usage and did not improve performance.

### 4. Tensor Core and Precision Benchmark

| Precision | Execution Time | Throughput |
|---|---:|---:|
| FP32 | 6.0275 ms | 2.850 TFLOPS |
| TF32 setting | 4.1093 ms | 4.181 TFLOPS |
| FP16 | 0.4868 ms | 35.293 TFLOPS |

FP16 delivered approximately 12.4× the FP32 throughput in this benchmark.

**Hardware note:** Tesla T4 supports FP16 Tensor Core acceleration but does not have native TF32 Tensor Core support. The TF32 setting result should not be interpreted as native TF32 Tensor Core execution.

## Final Experiment: Conventional vs GPU-aware Training

The final experiment compared two ResNet-50 training approaches under the same dataset and training configuration.

### Dataset and Model

- Model: ResNet-50
- Dataset: CIFAR-10 subset
- Training samples: 10,000
- Validation samples: 2,000
- Input resolution: 224 × 224
- Batch size: 64
- Epochs: 2
- Optimizer: AdamW
- GPU: NVIDIA Tesla T4

Both approaches started from the same initial model weights.

### Approach 1 — Conventional Training

- FP32 precision
- Contiguous memory format
- Pageable CPU memory
- Blocking data transfers
- `num_workers=0`
- Conventional gradient reset

### Approach 2 — GPU-aware Training

- FP16 mixed precision
- Channels-last memory format
- Pinned CPU memory
- Non-blocking data transfers
- `num_workers=2`
- `optimizer.zero_grad(set_to_none=True)`

### Results

| Metric | Conventional | GPU-aware |
|---|---:|---:|
| Epoch 1 loss | 1.6913 | 1.9082 |
| Epoch 2 loss | 1.2799 | 1.5105 |
| Training time | 241.80 sec | 64.31 sec |
| Throughput | 82.71 images/sec | 311.01 images/sec |
| Peak GPU memory | 6.11 GB | 3.57 GB |
| Validation accuracy | 34.75% | 38.15% |

The GPU-aware approach significantly reduced training time and memory consumption in this experiment.

The improvement comes from combining multiple optimizations rather than relying on a single technique.

## Key Learnings

Through these experiments, I explored:

- GPU architecture: SMs, CUDA cores, warps and Tensor Cores.
- Parallel execution: CUDA blocks, thread scheduling and SM mapping.
- Memory architecture: VRAM, cache policies, pinned memory and coalescing.
- GPU programming: Triton kernels and tiled matrix multiplication.
- Numerical precision: FP32, FP16 and TF32 configuration.
- Performance engineering: throughput, latency, memory allocation and profiling.
- Deep learning optimization: mixed precision, channels-last format and efficient data loading.

## Limitations

- Experiments were conducted using Kaggle GPU instances.
- Most microbenchmarks were performed on a single Tesla T4.
- The final comparison used only two training epochs and a subset of CIFAR-10.
- The final speedup represents one benchmark run; repeated runs and ablation studies are required for stronger conclusions.
- The validation accuracy difference does not establish that GPU optimizations inherently improve model quality.
- Some measurements represent effective benchmark throughput rather than direct hardware-counter measurements.

## Future Improvements

- Perform repeated benchmark runs with controlled random seeds.
- Conduct ablation studies to isolate individual optimization contributions.
- Profile GPU utilization during training.
- Explore multi-GPU training and communication overhead.
- Evaluate performance on larger datasets and models.
- Investigate CUDA kernels and advanced Triton optimizations.

## Conclusion

This project started with understanding how GPUs execute parallel workloads and progressed toward applying that knowledge to real deep learning training.

The experiments demonstrated that understanding memory transfers, parallel execution, memory access patterns, precision and GPU resource utilization can lead to meaningful performance improvements.

The final benchmark achieved a **3.76× training speedup** and **41.6% lower peak GPU memory usage** using a GPU-aware training configuration.

**The central takeaway: GPU performance is not just about using powerful hardware; it is about understanding how to use that hardware efficiently.**

---

*Built and experimentally evaluated using PyTorch, Triton, CUDA and NVIDIA Tesla T4 GPUs.*
