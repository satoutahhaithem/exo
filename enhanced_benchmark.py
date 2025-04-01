#!/usr/bin/env python3
import time
import psutil
import numpy as np
import asyncio
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from datetime import datetime
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

class ModelBenchmark:
    def __init__(self, models=None, prompts=None, inference_engine="tinygrad", runs=3):
        """Initialize the benchmarking class with models, prompts, and settings."""
        self.models = models if models else MODELS_TO_TEST
        self.prompts = prompts if prompts else TEST_PROMPTS
        self.inference_engine = inference_engine
        self.runs = runs
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    async def benchmark_model(self, model_id, prompt):
        """Benchmark a single model with a single prompt."""
        print(f"\nBenchmarking model: {model_id}")
        print(f"Prompt: {prompt[:50]}...")
        
        try:
            # Create shard and downloader
            shard = Shard(model_id=model_id)
            shard_downloader = ShardDownloader()
            
            # Get inference engine
            engine = get_inference_engine(self.inference_engine, shard_downloader)
            
            # Measure memory before
            mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            # Measure load time
            load_start_time = time.time()
            
            # Encode prompt
            tokens = await engine.encode(shard, prompt)
            
            # Run inference
            inference_start_time = time.time()
            load_time = inference_start_time - load_start_time
            
            x = tokens.reshape(1, -1)
            output_data, _ = await engine.infer_tensor("benchmark", shard, x)
            
            # Decode output
            result = await engine.decode(shard, output_data)
            end_time = time.time()
            
            # Calculate times
            inference_time = end_time - inference_start_time
            total_time = end_time - load_start_time
            
            # Measure memory after
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_usage = mem_after - mem_before
            
            # Calculate tokens per second
            output_tokens = len(result.split())
            tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0
            
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            
            result_data = {
                "model": model_id,
                "prompt": prompt,
                "load_time": load_time,
                "inference_time": inference_time,
                "total_time": total_time,
                "memory_usage": memory_usage,
                "output_length": len(result),
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
                "cpu_percent": cpu_percent,
                "success": True
            }
            
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Inference time: {inference_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Memory usage: {memory_usage:.2f}MB")
            print(f"  Tokens per second: {tokens_per_second:.2f}")
            
            return result_data
            
        except Exception as e:
            print(f"  Error benchmarking {model_id}: {e}")
            return {
                "model": model_id,
                "prompt": prompt,
                "load_time": 0,
                "inference_time": 0,
                "total_time": 0,
                "memory_usage": 0,
                "output_length": 0,
                "output_tokens": 0,
                "tokens_per_second": 0,
                "cpu_percent": 0,
                "success": False,
                "error": str(e)
            }

    async def run_benchmarks(self):
        """Run benchmarks for all models and prompts."""
        for model in self.models:
            for prompt in self.prompts:
                for run in range(self.runs):
                    print(f"Run {run+1}/{self.runs}")
                    result = await self.benchmark_model(model, prompt)
                    result["run"] = run + 1
                    self.results.append(result)
        
        return self.results
    
    def save_results(self, filename=None):
        """Save benchmark results to a CSV file."""
        if not filename:
            filename = f"benchmark_results_{self.timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return filename
    
    def generate_comparison_table(self):
        """Generate a comparison table of model performance."""
        if not self.results:
            print("No benchmark results available.")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Filter successful runs
        df_success = df[df["success"] == True]
        
        if df_success.empty:
            print("No successful benchmark runs to analyze.")
            return None
        
        # Group by model and calculate averages
        model_metrics = df_success.groupby("model").agg({
            "load_time": "mean",
            "inference_time": "mean",
            "total_time": "mean",
            "memory_usage": "mean",
            "tokens_per_second": "mean",
            "cpu_percent": "mean"
        }).reset_index()
        
        # Round values for better display
        for col in model_metrics.columns:
            if col != "model":
                model_metrics[col] = model_metrics[col].round(2)
        
        # Create a formatted table
        table_data = []
        for _, row in model_metrics.iterrows():
            table_data.append({
                "Model": row["model"],
                "Load Time (s)": row["load_time"],
                "Inference Time (s)": row["inference_time"],
                "Total Time (s)": row["total_time"],
                "Memory (MB)": row["memory_usage"],
                "Tokens/Second": row["tokens_per_second"],
                "CPU (%)": row["cpu_percent"]
            })
        
        # Create a formatted table string
        table_str = tabulate(table_data, headers="keys", tablefmt="grid")
        print("\nModel Performance Comparison:")
        print(table_str)
        
        # Save the table to a file
        table_file = f"model_comparison_{self.timestamp}.txt"
        with open(table_file, "w") as f:
            f.write("Model Performance Comparison\n")
            f.write("===========================\n\n")
            f.write(table_str)
            f.write("\n\nBenchmark conducted on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write("\nNumber of runs per model-prompt combination: " + str(self.runs))
            f.write("\nInference engine: " + self.inference_engine)
        
        print(f"Comparison table saved to {table_file}")
        return table_file
    
    def visualize_results(self):
        """Create visualizations of benchmark results."""
        if not self.results:
            print("No benchmark results available.")
            return []
        
        df = pd.DataFrame(self.results)
        
        # Filter successful runs
        df_success = df[df["success"] == True]
        
        if df_success.empty:
            print("No successful benchmark runs to analyze.")
            return []
        
        # Create output directory if it doesn't exist
        viz_dir = "benchmark_visualizations_" + self.timestamp
        os.makedirs(viz_dir, exist_ok=True)
        
        saved_files = []
        
        # Average latency by model
        plt.figure(figsize=(12, 6))
        avg_latency = df_success.groupby('model')['inference_time'].mean().sort_values()
        avg_latency.plot(kind='bar')
        plt.title('Average Inference Latency by Model')
        plt.ylabel('Latency (seconds)')
        plt.tight_layout()
        latency_file = f"{viz_dir}/latency_by_model.png"
        plt.savefig(latency_file)
        saved_files.append(latency_file)
        
        # Average memory usage by model
        plt.figure(figsize=(12, 6))
        avg_memory = df_success.groupby('model')['memory_usage'].mean().sort_values()
        avg_memory.plot(kind='bar')
        plt.title('Average Memory Usage by Model')
        plt.ylabel('Memory Usage (MB)')
        plt.tight_layout()
        memory_file = f"{viz_dir}/memory_by_model.png"
        plt.savefig(memory_file)
        saved_files.append(memory_file)
        
        # Tokens per second by model
        plt.figure(figsize=(12, 6))
        avg_tps = df_success.groupby('model')['tokens_per_second'].mean().sort_values()
        avg_tps.plot(kind='bar')
        plt.title('Average Tokens per Second by Model')
        plt.ylabel('Tokens per Second')
        plt.tight_layout()
        tps_file = f"{viz_dir}/tps_by_model.png"
        plt.savefig(tps_file)
        saved_files.append(tps_file)
        
        # Load time vs Inference time
        plt.figure(figsize=(12, 6))
        models = df_success['model'].unique()
        load_times = [df_success[df_success['model'] == model]['load_time'].mean() for model in models]
        inference_times = [df_success[df_success['model'] == model]['inference_time'].mean() for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, load_times, width, label='Load Time')
        plt.bar(x + width/2, inference_times, width, label='Inference Time')
        
        plt.xlabel('Model')
        plt.ylabel('Time (seconds)')
        plt.title('Load Time vs Inference Time by Model')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        time_comparison_file = f"{viz_dir}/load_vs_inference_time.png"
        plt.savefig(time_comparison_file)
        saved_files.append(time_comparison_file)
        
        print(f"Visualizations saved to {viz_dir}/")
        return saved_files

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark LLM models in Exo')
    parser.add_argument('--models', nargs='+', help='List of models to benchmark')
    parser.add_argument('--prompts', nargs='+', help='List of prompts to use')
    parser.add_argument('--engine', default='tinygrad', help='Inference engine to use')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per model-prompt combination')
    parser.add_argument('--output', help='Output file for results (CSV)')
    parser.add_argument('--no-visualize', action='store_true', help='Skip visualization generation')
    return parser.parse_args()

async def main():
    """Main function to run benchmarks."""
    args = parse_arguments()
    
    # Initialize benchmarking
    benchmark = ModelBenchmark(
        models=args.models,
        prompts=args.prompts,
        inference_engine=args.engine,
        runs=args.runs
    )
    
    # Run benchmarks
    print("Starting benchmarks...")
    results = await benchmark.run_benchmarks()
    
    # Save results
    output_file = args.output if args.output else None
    benchmark.save_results(output_file)
    
    # Generate comparison table
    benchmark.generate_comparison_table()
    
    # Create visualizations
    if not args.no_visualize:
        benchmark.visualize_results()
    
    print("Benchmarking complete!")

if __name__ == "__main__":
    asyncio.run(main())
