"""
Compare and visualize benchmark results between vLLM and TGI
"""

import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def load_results(filepath: str) -> dict:
    """Load benchmark results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_throughput(vllm_results: dict, tgi_results: dict):
    """Compare throughput between vLLM and TGI"""
    vllm_throughput = vllm_results['results']['throughput_tokens_per_sec']
    tgi_throughput = tgi_results['results']['throughput_tokens_per_sec']
    
    speedup = vllm_throughput / tgi_throughput
    
    print("\n" + "="*60)
    print("THROUGHPUT COMPARISON")
    print("="*60)
    print(f"vLLM:  {vllm_throughput:,.2f} tokens/sec")
    print(f"TGI:   {tgi_throughput:,.2f} tokens/sec")
    print(f"Speedup: {speedup:.2f}x")
    print("="*60)
    
    return vllm_throughput, tgi_throughput, speedup


def compare_latency(vllm_results: dict, tgi_results: dict):
    """Compare latency metrics between vLLM and TGI"""
    print("\n" + "="*60)
    print("LATENCY COMPARISON")
    print("="*60)
    
    metrics = ['mean_latency', 'p50_latency', 'p95_latency', 'p99_latency', 'mean_ttft', 'mean_tpot']
    metric_names = ['Mean Latency', 'p50 Latency', 'p95 Latency', 'p99 Latency', 'Mean TTFT', 'Mean TPOT']
    
    for metric, name in zip(metrics, metric_names):
        vllm_val = vllm_results['results'][metric]
        tgi_val = tgi_results['results'][metric]
        diff = ((vllm_val - tgi_val) / tgi_val) * 100
        
        print(f"\n{name}:")
        print(f"  vLLM: {vllm_val:.3f}s")
        print(f"  TGI:  {tgi_val:.3f}s")
        print(f"  Diff: {diff:+.1f}%")
    
    print("="*60)


def compare_memory(vllm_results: dict, tgi_results: dict):
    """Compare GPU memory usage"""
    vllm_mem = vllm_results['results']['gpu_memory_used_gb']
    tgi_mem = tgi_results['results']['gpu_memory_used_gb']
    savings = ((tgi_mem - vllm_mem) / tgi_mem) * 100
    
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    print(f"vLLM: {vllm_mem:.1f} GB")
    print(f"TGI:  {tgi_mem:.1f} GB")
    print(f"Memory Savings: {savings:.1f}%")
    print("="*60)


def create_visualizations(vllm_results: dict, tgi_results: dict, output_dir: str):
    """Create comparison visualizations"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    # 1. Throughput comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    frameworks = ['vLLM', 'TGI']
    throughputs = [
        vllm_results['results']['throughput_tokens_per_sec'],
        tgi_results['results']['throughput_tokens_per_sec']
    ]
    
    bars = ax.bar(frameworks, throughputs, color=['#2E86AB', '#A23B72'])
    ax.set_ylabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_title('Throughput Comparison: vLLM vs TGI', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / 'throughput_comparison.png', dpi=300)
    print(f"\nSaved: {output_path / 'throughput_comparison.png'}")
    
    # 2. Latency comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    latency_metrics = ['Mean', 'p50', 'p95', 'p99']
    vllm_latencies = [
        vllm_results['results']['mean_latency'],
        vllm_results['results']['p50_latency'],
        vllm_results['results']['p95_latency'],
        vllm_results['results']['p99_latency']
    ]
    tgi_latencies = [
        tgi_results['results']['mean_latency'],
        tgi_results['results']['p50_latency'],
        tgi_results['results']['p95_latency'],
        tgi_results['results']['p99_latency']
    ]
    
    x = np.arange(len(latency_metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, vllm_latencies, width, label='vLLM', color='#2E86AB')
    bars2 = ax.bar(x + width/2, tgi_latencies, width, label='TGI', color='#A23B72')
    
    ax.set_xlabel('Latency Metric', fontsize=12)
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Latency Distribution: vLLM vs TGI', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(latency_metrics)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'latency_comparison.png', dpi=300)
    print(f"Saved: {output_path / 'latency_comparison.png'}")
    
    # 3. Memory usage comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    frameworks = ['vLLM', 'TGI']
    memory_usage = [
        vllm_results['results']['gpu_memory_used_gb'],
        tgi_results['results']['gpu_memory_used_gb']
    ]
    
    bars = ax.bar(frameworks, memory_usage, color=['#2E86AB', '#A23B72'])
    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_title('GPU Memory Usage: vLLM vs TGI', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} GB',
                ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path / 'memory_comparison.png', dpi=300)
    print(f"Saved: {output_path / 'memory_comparison.png'}")
    
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description="Compare vLLM and TGI benchmark results")
    parser.add_argument("--vllm-results", type=str, required=True,
                       help="Path to vLLM results JSON")
    parser.add_argument("--tgi-results", type=str, required=True,
                       help="Path to TGI results JSON")
    parser.add_argument("--output-dir", type=str, default="results/visualizations",
                       help="Directory for output visualizations")
    
    args = parser.parse_args()
    
    # Load results
    print("Loading results...")
    vllm_results = load_results(args.vllm_results)
    tgi_results = load_results(args.tgi_results)
    
    # Compare metrics
    compare_throughput(vllm_results, tgi_results)
    compare_latency(vllm_results, tgi_results)
    compare_memory(vllm_results, tgi_results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(vllm_results, tgi_results, args.output_dir)
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
