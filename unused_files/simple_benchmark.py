import time
import subprocess
from tabulate import tabulate
from tqdm import tqdm

def benchmark_model(model_name, prompt, runs=3):
    """
    Benchmarks a given model by running the specified prompt multiple times.
    Returns the average latency in seconds.
    """
    latencies = []
    print(f"\nTesting model: {model_name}")
    print(f"Prompt: {prompt[:30]}...")
    
    for i in tqdm(range(runs), desc=f"{model_name} runs", unit="run"):
        start_time = time.time()
        result = subprocess.run(
            ["exo", "run", model_name, "--prompt", prompt],
            capture_output=True,
            text=True
        )
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage Latency for {model_name}: {avg_latency:.2f} seconds")
    print("-" * 40)
    return avg_latency

def main():
    models = ["llama-3.1-8b", "llama-3.2-1b", "mistral-7b"]
    prompt = "Explain the concept of artificial intelligence in simple terms."
    results = []

    for model in models:
        avg_latency = benchmark_model(model, prompt, runs=3)
        results.append({"Model": model, "Average Latency (s)": f"{avg_latency:.2f}"})
    
    print("\nBenchmark Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    main()
