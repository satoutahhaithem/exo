import time
import psutil
import numpy as np
import asyncio
import os
from exo.download.shard_download import ShardDownloader
from exo.inference import get_inference_engine
from exo.inference.shard import Shard

# List of models to benchmark
MODELS_TO_TEST = [
    "llama-3.1-8b",
    "llama-3.2-1b",
    "mistral-7b",
    "qwen-1.5-7b",
    "deepseek-r1"
]

# Standard test prompts
TEST_PROMPTS = [
    "Explain the concept of artificial intelligence in simple terms.",
    "Write a short poem about technology.",
    "Summarize the key benefits of renewable energy.",
    "What are the main differences between classical and quantum computing?",
    "Describe the process of photosynthesis in plants."
]

async def benchmark_model(model_id, prompt, inference_engine_name="tinygrad"):
    """Benchmark a single model with a single prompt."""
    # Create shard and downloader
    shard = Shard(model_id=model_id)
    shard_downloader = ShardDownloader()
    
    # Get inference engine
    engine = get_inference_engine(inference_engine_name, shard_downloader)
    
    # Measure memory before
    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # Encode prompt
    start_time = time.time()
    tokens = await engine.encode(shard, prompt)
    
    # Run inference
    x = tokens.reshape(1, -1)
    output_data, _ = await engine.infer_tensor("benchmark", shard, x)
    
    # Decode output
    result = await engine.decode(shard, output_data)
    end_time = time.time()
    
    # Measure memory after
    mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    return {
        "model": model_id,
        "prompt": prompt,
        "latency": end_time - start_time,
        "memory_usage": mem_after - mem_before,
        "output_length": len(result),
        "tokens_per_second": len(result) / (end_time - start_time)
    }

async def run_benchmarks():
    """Run benchmarks for all models and prompts."""
    results = []
    
    for model in MODELS_TO_TEST:
        print(f"Benchmarking model: {model}")
        for prompt in TEST_PROMPTS:
            print(f"  Testing prompt: {prompt[:30]}...")
            try:
                result = await benchmark_model(model, prompt)
                results.append(result)
                print(f"    Latency: {result['latency']:.2f}s, Memory: {result['memory_usage']:.2f}MB")
            except Exception as e:
                print(f"    Error benchmarking {model}: {e}")
    
    return results

def save_results(results, filename="benchmark_results.csv"):
    """Save benchmark results to a CSV file."""
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def visualize_results(results):
    """Create visualizations of benchmark results."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # Average latency by model
    plt.figure(figsize=(12, 6))
    avg_latency = df.groupby('model')['latency'].mean().sort_values()
    avg_latency.plot(kind='bar')
    plt.title('Average Inference Latency by Model')
    plt.ylabel('Latency (seconds)')
    plt.tight_layout()
    plt.savefig('latency_by_model.png')
    
    # Average memory usage by model
    plt.figure(figsize=(12, 6))
    avg_memory = df.groupby('model')['memory_usage'].mean().sort_values()
    avg_memory.plot(kind='bar')
    plt.title('Average Memory Usage by Model')
    plt.ylabel('Memory Usage (MB)')
    plt.tight_layout()
    plt.savefig('memory_by_model.png')
    
    # Tokens per second by model
    plt.figure(figsize=(12, 6))
    avg_tps = df.groupby('model')['tokens_per_second'].mean().sort_values()
    avg_tps.plot(kind='bar')
    plt.title('Average Tokens per Second by Model')
    plt.ylabel('Tokens per Second')
    plt.tight_layout()
    plt.savefig('tps_by_model.png')
    
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    # Run benchmarks
    results = asyncio.run(run_benchmarks())
    
    # Save results
    save_results(results)
    
    # Create visualizations
    try:
        visualize_results(results)
    except ImportError:
        print("Matplotlib and/or pandas not installed. Skipping visualizations.")
    
    # Print summary
    print("\nBenchmark Summary:")
    for model in MODELS_TO_TEST:
        model_results = [r for r in results if r["model"] == model]
        if model_results:
            avg_latency = np.mean([r["latency"] for r in model_results])
            avg_memory = np.mean([r["memory_usage"] for r in model_results])
            avg_tps = np.mean([r["tokens_per_second"] for r in model_results])
            print(f"{model}: Avg Latency: {avg_latency:.2f}s, Avg Memory: {avg_memory:.2f}MB, Avg TPS: {avg_tps:.2f}")
