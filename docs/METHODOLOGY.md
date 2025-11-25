# Benchmarking Methodology

This document provides a comprehensive overview of the methodology used in our comparative study of vLLM and HuggingFace TGI.

## Table of Contents

- [Experimental Setup](#experimental-setup)
- [Hardware Configuration](#hardware-configuration)
- [Software Versions](#software-versions)
- [Workload Design](#workload-design)
- [Metrics Collection](#metrics-collection)
- [Statistical Analysis](#statistical-analysis)
- [Reproducibility](#reproducibility)

## Experimental Setup

### Hardware Configuration

**GPU Infrastructure:**
- 4x NVIDIA A100 80GB GPUs
- NVLink for GPU-to-GPU communication
- PCIe Gen4 connectivity

**Compute Resources:**
- CPU: AMD EPYC 7763 64-Core Processor
- RAM: 512GB DDR4
- Storage: NVMe SSD (2TB)

**Network:**
- 10 Gbps Ethernet
- Low-latency local network for benchmarking

### Software Versions

**Frameworks:**
- vLLM: v0.6.1
- HuggingFace TGI: v2.3.0

**Supporting Libraries:**
- CUDA: 12.1
- PyTorch: 2.1.0
- Transformers: 4.35.0
- Python: 3.10

**Operating System:**
- Ubuntu 22.04 LTS
- Kernel: 5.15.0

## Workload Design

### Concurrency Levels

We tested three distinct concurrency scenarios to represent different production use cases:

#### Low Concurrency (1-10 concurrent users)
- **Use Case:** Interactive chatbot, single-user applications
- **Test Points:** 1, 5, 10 concurrent requests
- **Expected Behavior:** Focus on latency optimization

#### Medium Concurrency (10-50 users)
- **Use Case:** Typical production API load, small team deployment
- **Test Points:** 10, 25, 50 concurrent requests
- **Expected Behavior:** Balance between throughput and latency

#### High Concurrency (50-200 users)
- **Use Case:** Multi-tenant serving, high-traffic API endpoints
- **Test Points:** 50, 100, 150, 200 concurrent requests
- **Expected Behavior:** Maximum throughput optimization

### Input/Output Characteristics

#### Prompt Length Distribution
- **Distribution:** Poisson distribution
- **Mean:** 512 tokens
- **Range:** 128-2048 tokens
- **Rationale:** Represents real-world conversation lengths from ShareGPT dataset

#### Generation Length Distribution
- **Distribution:** Poisson distribution
- **Mean:** 256 tokens
- **Range:** 64-512 tokens
- **Rationale:** Typical AI assistant response lengths

#### Sampling Parameters
```python
{
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 512,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}
```

### Dataset

**Source:** ShareGPT conversations
- Real user-AI interactions from production systems
- Diverse topics and conversation styles
- Pre-filtered for quality and appropriateness

**Dataset Size:**
- Total prompts: 10,000+
- Selected for benchmarks: 1,000 per run
- Validation set: 100 (not used in benchmarks)

**Data Preprocessing:**
- Tokenization using LLaMA tokenizer
- Filtering for length constraints
- Deduplication of similar prompts

## Experimental Protocol

### Benchmark Execution Flow

#### 1. System Initialization
```bash
# Clean GPU memory
nvidia-smi --gpu-reset

# Start serving system
# (vLLM or TGI specific commands)

# Wait for system ready (health check)
```

#### 2. Warm-up Phase
- **Duration:** 100 requests
- **Purpose:** Stabilize GPU state, populate caches
- **Metrics:** Not recorded
- **Verification:** Ensure no errors or crashes

#### 3. Measurement Phase
- **Duration:** 1,000 requests
- **Metrics Collection:** Full instrumentation enabled
- **Logging:** Per-request timing and resource metrics
- **Monitoring:** GPU utilization, memory usage

#### 4. Cool-down Phase
- **Duration:** 30 seconds
- **Purpose:** Allow GPU to stabilize between experiments
- **Actions:** No requests sent, metrics recorded for baseline

#### 5. Repetition
- **Number of runs:** 3 per configuration
- **Aggregation:** Report median values
- **Variance:** Calculate standard deviation for error bars

### Configuration Matrix

**Models Tested:**
- LLaMA-2-7B (single GPU)
- LLaMA-2-13B (single GPU)
- LLaMA-2-70B (4-GPU tensor parallel)

**Concurrency Levels:**
- [1, 5, 10, 25, 50, 100, 150, 200]

**Total Experiments:**
- 3 models × 8 concurrency levels × 2 frameworks × 3 repetitions = 144 experiments

## Metrics Collection

### Throughput Metrics

#### Tokens per Second
```python
total_tokens_generated / total_wall_clock_time
```
- Includes both prompt processing and generation
- Measured across all concurrent requests
- Primary metric for batch processing workloads

#### Requests per Second
```python
total_requests_completed / total_wall_clock_time
```
- Completion rate of requests
- Useful for understanding system capacity

### Latency Metrics

#### Time to First Token (TTFT)
```python
timestamp(first_token_received) - timestamp(request_sent)
```
- Critical for interactive applications
- Includes prompt processing time
- Measured per request, aggregated by percentiles

#### Time per Output Token (TPOT)
```python
(timestamp(last_token) - timestamp(first_token)) / num_output_tokens
```
- Generation speed per token
- Indicates decoding efficiency
- Lower is better

#### End-to-End Latency
```python
timestamp(request_complete) - timestamp(request_sent)
```
- Total request completion time
- Includes TTFT + generation time
- User-perceived latency

#### Percentile Analysis
- **p50 (median):** Typical user experience
- **p95:** Near-worst-case for most users
- **p99:** Worst-case tail latency
- **p99.9:** Extreme outliers

### Resource Metrics

#### GPU Memory Usage
```bash
# Collected via nvidia-smi every 100ms
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```
- **Peak memory:** Maximum allocation during benchmark
- **Average memory:** Mean usage across measurement phase
- **Memory efficiency:** tokens_generated / GB_used

#### GPU Utilization
```bash
# SM (Streaming Multiprocessor) utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv
```
- Percentage of time GPU is actively computing
- Target: >90% for throughput workloads
- Measured every 100ms, averaged

#### CPU and RAM
- CPU usage per core (via psutil)
- Total RAM consumption
- Swap usage (should be 0)

## Statistical Analysis

### Significance Testing

**Student's t-test:**
- Null hypothesis: No difference between vLLM and TGI
- Significance level: α = 0.05
- Applied to throughput and latency metrics

**Effect Size:**
- Cohen's d for practical significance
- Large effect: d > 0.8

### Error Bars

All plots include error bars representing:
- **1 standard deviation** for mean values
- **Interquartile range** for median values

### Outlier Detection

**IQR Method:**
```python
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
outliers = values < (Q1 - 1.5*IQR) or values > (Q3 + 1.5*IQR)
```

Outliers are reported but not excluded from analysis.

## Reproducibility

### Environment Specification

**Docker Containers:**
```bash
# vLLM container
docker run --gpus all \
  --shm-size 10g \
  vllm/vllm-openai:v0.6.1

# TGI container
docker run --gpus all \
  --shm-size 10g \
  ghcr.io/huggingface/text-generation-inference:2.3.0
```

### Configuration Files

All benchmark configurations are versioned in:
- `benchmarks/config.py`: Framework-specific settings
- `benchmarks/workload_config.json`: Request distributions
- `scripts/env_setup.sh`: Environment variables

### Random Seed Control

```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### Data Availability

- Raw results: `results/*.json`
- Processed data: `results/processed/*.csv`
- Visualizations: `results/visualizations/*.png`

## Limitations and Considerations

### Known Limitations

1. **Hardware Specific:** Results are specific to A100 GPUs
2. **Model Specific:** LLaMA-2 architecture only
3. **Workload Specific:** ShareGPT conversation style
4. **Version Specific:** Framework versions as of Nov 2025

### Factors Not Tested

- Quantization (INT8, INT4)
- Speculative decoding
- Multi-node distributed inference
- Long context (>4096 tokens)
- Fine-tuned models

### Potential Confounding Variables

- OS scheduler variations
- Background processes
- GPU thermal throttling
- Network jitter (minimal in local setup)

**Mitigation:** Repeated measurements, system monitoring, controlled environment

## References

1. vLLM Documentation: https://docs.vllm.ai/
2. TGI Documentation: https://huggingface.co/docs/text-generation-inference/
3. ShareGPT Dataset: https://sharegpt.com/
4. NVIDIA A100 Specifications: https://www.nvidia.com/en-us/data-center/a100/

---

For questions about methodology, please open an issue or contact the author.
