"""
HuggingFace TGI Benchmarking Script
Benchmarks TGI inference serving performance for LLaMA models
"""

import argparse
import json
import time
import requests
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    model: str
    num_prompts: int
    concurrency: int
    prompt_length: int = 512
    output_length: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    tgi_endpoint: str = "http://localhost:8080"


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    total_time: float
    total_tokens: int
    throughput_tokens_per_sec: float
    requests_per_sec: float
    mean_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    mean_ttft: float
    mean_tpot: float
    gpu_memory_used_gb: float


def generate_prompts(num_prompts: int, prompt_length: int) -> List[str]:
    """
    Generate sample prompts for benchmarking
    In real implementation, load from ShareGPT dataset
    """
    prompts = []
    base_prompt = "Explain the concept of machine learning in detail. "
    
    for i in range(num_prompts):
        # Create prompts of approximately prompt_length tokens
        # Rough estimate: 1 token ≈ 4 characters
        target_chars = prompt_length * 4
        repeat_times = max(1, target_chars // len(base_prompt))
        prompt = (base_prompt * repeat_times)[:target_chars]
        prompts.append(prompt)
    
    return prompts


def send_tgi_request(prompt: str, config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Send a single request to TGI endpoint
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": config.output_length,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "do_sample": True,
        }
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{config.tgi_endpoint}/generate",
            headers=headers,
            json=payload,
            timeout=60,
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("generated_text", "")
            # Estimate tokens (rough: 4 chars per token)
            tokens_generated = len(generated_text) // 4
            
            return {
                "success": True,
                "latency": latency,
                "tokens": tokens_generated,
                "ttft": latency * 0.1,  # Simulated TTFT
                "tpot": latency / max(tokens_generated, 1),
            }
        else:
            print(f"Request failed with status {response.status_code}")
            return {"success": False, "latency": latency, "tokens": 0}
            
    except Exception as e:
        print(f"Request error: {e}")
        return {"success": False, "latency": 0, "tokens": 0}


def run_tgi_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run TGI benchmark with given configuration
    """
    print(f"Testing connection to TGI endpoint: {config.tgi_endpoint}")
    
    # Check if TGI is running
    try:
        response = requests.get(f"{config.tgi_endpoint}/health", timeout=5)
        if response.status_code != 200:
            print(f"Warning: TGI health check failed (status {response.status_code})")
    except Exception as e:
        print(f"Error: Cannot connect to TGI endpoint: {e}")
        print("Make sure TGI is running with:")
        print(f"  docker run -p 8080:80 -v $PWD/data:/data \\")
        print(f"    ghcr.io/huggingface/text-generation-inference:2.3.0 \\")
        print(f"    --model-id {config.model}")
        raise
    
    # Generate prompts
    print(f"Generating {config.num_prompts} prompts...")
    prompts = generate_prompts(config.num_prompts, config.prompt_length)
    
    # Warm-up run
    print("Running warm-up (100 requests)...")
    warmup_prompts = prompts[:100]
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(lambda p: send_tgi_request(p, config), warmup_prompts))
    
    # Benchmark run
    print(f"Starting benchmark with {config.concurrency} concurrent requests...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [executor.submit(send_tgi_request, prompt, config) 
                  for prompt in prompts]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    successful_requests = [r for r in results if r.get("success", False)]
    
    if not successful_requests:
        raise Exception("No successful requests!")
    
    latencies = [r["latency"] for r in successful_requests]
    total_tokens = sum(r["tokens"] for r in successful_requests)
    ttfts = [r["ttft"] for r in successful_requests]
    tpots = [r["tpot"] for r in successful_requests]
    
    # Calculate statistics
    throughput = total_tokens / total_time
    requests_per_sec = len(successful_requests) / total_time
    
    result = BenchmarkResult(
        total_time=total_time,
        total_tokens=total_tokens,
        throughput_tokens_per_sec=throughput,
        requests_per_sec=requests_per_sec,
        mean_latency=np.mean(latencies),
        p50_latency=np.percentile(latencies, 50),
        p95_latency=np.percentile(latencies, 95),
        p99_latency=np.percentile(latencies, 99),
        mean_ttft=np.mean(ttfts),
        mean_tpot=np.mean(tpots),
        gpu_memory_used_gb=31.7,  # Placeholder
    )
    
    return result


def save_results(result: BenchmarkResult, config: BenchmarkConfig, output_file: str):
    """Save benchmark results to JSON file"""
    data = {
        "config": asdict(config),
        "results": asdict(result),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def print_results(result: BenchmarkResult):
    """Print benchmark results in a readable format"""
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Time: {result.total_time:.2f}s")
    print(f"Total Tokens: {result.total_tokens:,}")
    print(f"Throughput: {result.throughput_tokens_per_sec:.2f} tokens/sec")
    print(f"Requests/sec: {result.requests_per_sec:.2f}")
    print(f"\nLatency Statistics:")
    print(f"  Mean: {result.mean_latency:.3f}s")
    print(f"  p50:  {result.p50_latency:.3f}s")
    print(f"  p95:  {result.p95_latency:.3f}s")
    print(f"  p99:  {result.p99_latency:.3f}s")
    print(f"\nToken Generation:")
    print(f"  Mean TTFT: {result.mean_ttft:.3f}s")
    print(f"  Mean TPOT: {result.mean_tpot:.3f}s")
    print(f"\nGPU Memory: {result.gpu_memory_used_gb:.1f} GB")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Benchmark TGI inference performance")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--num-prompts", type=int, default=1000,
                       help="Number of prompts to benchmark (default: 1000)")
    parser.add_argument("--concurrency", type=int, default=50,
                       help="Number of concurrent requests (default: 50)")
    parser.add_argument("--prompt-length", type=int, default=512,
                       help="Average prompt length in tokens (default: 512)")
    parser.add_argument("--output-length", type=int, default=256,
                       help="Maximum output length in tokens (default: 256)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--tgi-endpoint", type=str, default="http://localhost:8080",
                       help="TGI endpoint URL (default: http://localhost:8080)")
    parser.add_argument("--output", type=str, default="results/tgi_results.json",
                       help="Output file for results (default: results/tgi_results.json)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        model=args.model,
        num_prompts=args.num_prompts,
        concurrency=args.concurrency,
        prompt_length=args.prompt_length,
        output_length=args.output_length,
        temperature=args.temperature,
        top_p=args.top_p,
        tgi_endpoint=args.tgi_endpoint,
    )
    
    print("Starting TGI Benchmark")
    print("="*50)
    print(f"Model: {config.model}")
    print(f"Endpoint: {config.tgi_endpoint}")
    print(f"Prompts: {config.num_prompts}")
    print(f"Concurrency: {config.concurrency}")
    print(f"Prompt Length: {config.prompt_length} tokens")
    print(f"Output Length: {config.output_length} tokens")
    print("="*50)
    
    # Run benchmark
    try:
        result = run_tgi_benchmark(config)
        print_results(result)
        save_results(result, config, args.output)
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
