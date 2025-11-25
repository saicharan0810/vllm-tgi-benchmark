# vLLM vs HuggingFace TGI Benchmarking Study

A comprehensive performance evaluation of Large Language Model (LLM) inference serving systems.

[![arXiv](https://img.shields.io/badge/arXiv-2511.17593-b31b1b.svg)](https://arxiv.org/abs/2511.17593)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## 📄 Paper

**[Comparative Analysis of Large Language Model Inference Serving Systems: A Performance Study of vLLM and HuggingFace TGI](https://arxiv.org/abs/2511.17593)**

*Saicharan Kolluru*

Published on arXiv, November 2025

## 🎯 Overview

This repository contains the benchmarking code, analysis scripts, and results from our comprehensive study comparing [vLLM](https://github.com/vllm-project/vllm) and [HuggingFace Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) for serving Large Language Models.

### Key Findings

- **vLLM achieves 2-24x higher throughput** than TGI under high-concurrency workloads through its PagedAttention mechanism
- **TGI demonstrates 1.3-2x lower Time-to-First-Token (TTFT)** at low concurrency, better for interactive scenarios
- **vLLM utilizes 19-27% less GPU memory** through efficient memory management
- **vLLM achieves 85-92% GPU utilization** compared to TGI's 68-74%

## 📊 Benchmarked Systems

| Framework | Version | Key Features |
|-----------|---------|--------------|
| vLLM | v0.6.1 | PagedAttention, Continuous Batching |
| HuggingFace TGI | v2.3.0 | Dynamic Batching, Quantization Support |

**Models Tested:**
- LLaMA-2-7B (single GPU)
- LLaMA-2-13B (single GPU)
- LLaMA-2-70B (tensor parallel, 4 GPUs)

**Hardware:**
- 4x NVIDIA A100 80GB GPUs
- AMD EPYC 7763 64-Core Processor
- 512GB RAM

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.9+
python --version

# CUDA 12.1+
nvidia-smi

# Install dependencies
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vllm-tgi-benchmark.git
cd vllm-tgi-benchmark

# Install vLLM
pip install vllm==0.6.1

# Install TGI (Docker required)
docker pull ghcr.io/huggingface/text-generation-inference:2.3.0
```

### Running Benchmarks

```bash
# Benchmark vLLM
python benchmarks/run_vllm_benchmark.py \
    --model meta-llama/Llama-2-7b-hf \
    --num-prompts 1000 \
    --concurrency 50

# Benchmark TGI
python benchmarks/run_tgi_benchmark.py \
    --model meta-llama/Llama-2-7b-hf \
    --num-prompts 1000 \
    --concurrency 50

# Analyze results
python analysis/compare_results.py \
    --vllm-results results/vllm_results.json \
    --tgi-results results/tgi_results.json
```

## 📁 Repository Structure

```
vllm-tgi-benchmark/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── paper/
│   └── vllm_benchmarking_paper.pdf
├── benchmarks/
│   ├── run_vllm_benchmark.py  # vLLM benchmarking script
│   ├── run_tgi_benchmark.py   # TGI benchmarking script
│   ├── config.py              # Configuration settings
│   └── utils.py               # Utility functions
├── analysis/
│   ├── compare_results.py     # Result comparison script
│   ├── visualize.py           # Visualization tools
│   └── statistical_tests.py   # Statistical analysis
├── results/
│   ├── vllm_7b_results.json
│   ├── tgi_7b_results.json
│   ├── vllm_13b_results.json
│   ├── tgi_13b_results.json
│   └── visualizations/
│       ├── throughput_comparison.png
│       ├── latency_distribution.png
│       └── gpu_utilization.png
└── docs/
    ├── METHODOLOGY.md         # Detailed methodology
    ├── RESULTS.md             # Complete results
    └── FAQ.md                 # Frequently asked questions
```

## 📈 Key Metrics

Our benchmarks measure the following performance indicators:

### Throughput Metrics
- **Tokens per second** across all concurrent requests
- **Requests per second** completion rate
- **Effective GPU utilization**

### Latency Metrics
- **Time to First Token (TTFT)**: Time until first token generation
- **Time per Output Token (TPOT)**: Average generation time per token
- **End-to-end latency**: Total request completion time
- **Percentile analysis**: p50, p95, p99 latencies

### Resource Metrics
- **GPU memory usage** (peak and average)
- **GPU compute utilization** percentage
- **CPU and RAM usage**

## 🔬 Methodology

### Workload Design

**Concurrency Levels:**
- Low (1-10 concurrent users): Interactive chatbot simulation
- Medium (10-50 users): Typical production load
- High (50-200 users): Stress testing scenario

**Input/Output Characteristics:**
- Prompt lengths: Poisson distribution (mean=512 tokens)
- Generation lengths: Poisson distribution (mean=256 tokens)
- Sampling: Temperature=0.7, top-p=0.9

**Dataset:**
- ShareGPT conversations (real user-AI interactions)
- 1,000 requests per benchmark run
- 3 repetitions for statistical significance

### Experimental Protocol

1. **Warm-up**: 100 requests to stabilize system
2. **Measurement**: 1,000 requests with full metrics collection
3. **Cool-down**: 30 seconds between experiments
4. **Repetition**: 3 runs, report median values

See [METHODOLOGY.md](docs/METHODOLOGY.md) for complete details.

## 📊 Results Summary

### Throughput Performance

| Model | vLLM (tokens/sec) | TGI (tokens/sec) | Speedup |
|-------|-------------------|------------------|---------|
| LLaMA-2-7B @ 100 concurrent | 15,243 | 4,156 | 3.67x |
| LLaMA-2-13B @ 100 concurrent | 8,934 | 3,187 | 2.80x |
| LLaMA-2-70B @ 50 concurrent | 3,245 | 1,544 | 2.10x |

### Latency Performance (25 concurrent users)

| Model | Metric | vLLM | TGI |
|-------|--------|------|-----|
| LLaMA-2-7B | TTFT p50 | 0.24s | 0.18s |
| LLaMA-2-7B | Total Latency p50 | 4.82s | 5.91s |
| LLaMA-2-7B | TPOT p50 | 0.019s | 0.023s |

See [RESULTS.md](docs/RESULTS.md) for complete results and analysis.

## 🎯 Recommendations

### Use vLLM when:
- ✅ High throughput is critical (batch processing, offline evaluation)
- ✅ High concurrency (multi-tenant serving, API services)
- ✅ Memory constraints (larger models, longer contexts)
- ✅ GPU utilization matters (cost optimization)

### Use TGI when:
- ✅ Low TTFT is crucial (interactive chat, real-time assistance)
- ✅ Predictable latency needed (SLA-bound services)
- ✅ Quantization required (INT8/INT4 for resource constraints)
- ✅ HuggingFace ecosystem integration preferred

## 🔄 Reproducibility

All experiments are designed to be reproducible. We provide:

- ✅ Complete source code for benchmarks
- ✅ Configuration files with exact parameters
- ✅ Detailed environment specifications
- ✅ Statistical analysis scripts
- ✅ Raw result data (JSON format)

To reproduce our results:

```bash
# Run full benchmark suite
bash scripts/run_full_benchmark.sh

# Results will be saved to results/ directory
# Visualizations generated in results/visualizations/
```

## 📖 Citation

If you use this benchmark or find our work helpful, please cite:

```bibtex
@article{kolluru2025vllm,
  title={Comparative Analysis of Large Language Model Inference Serving Systems: A Performance Study of vLLM and HuggingFace TGI},
  author={Kolluru, Saicharan},
  journal={arXiv preprint arXiv:2511.17593},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs or issues
- Suggest improvements to benchmarking methodology
- Add support for additional frameworks
- Improve documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under CC BY 4.0 - see the [LICENSE](LICENSE) file for details.

The research paper is also available under CC BY 4.0 license.

## 🙏 Acknowledgments

- [vLLM Team](https://github.com/vllm-project/vllm) for developing PagedAttention
- [HuggingFace](https://huggingface.co/) for TGI and the model hub
- [Meta AI](https://ai.meta.com/) for releasing LLaMA-2 models

## 📬 Contact

**Saicharan Kolluru**
- Email: kscharan1608@gmail.com
- LinkedIn: [linkedin.com/in/kscharan1608](https://linkedin.com/in/kscharan1608)
- arXiv: [Author page](https://arxiv.org/a/kolluru_s_1)

## 🔗 Related Work

- [vLLM: Easy, Fast, and Cheap LLM Serving](https://vllm.ai/)
- [HuggingFace TGI Documentation](https://huggingface.co/docs/text-generation-inference)
- [LLaMA-2 Model Family](https://ai.meta.com/llama/)
- [Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)

## ⚠️ Notes

- This benchmark focuses on inference performance, not training
- Results may vary based on hardware, model versions, and workload characteristics
- Both frameworks are actively developed; performance may change in newer versions
- For production deployment, always benchmark with your specific use case

---

**Star ⭐ this repository if you find it helpful!**

Last updated: November 2025
