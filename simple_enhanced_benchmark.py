#!/usr/bin/env python3
import time
import subprocess
import pandas as pd
from tabulate import tabulate
import argparse
import os
from datetime import datetime

# List of models to benchmark
DEFAULT_MODELS = [
    "llama-3.1-8b",
    "llama-3.2-1b",
    "mistral-7b",
    "qwen-1.5-7b",
    "deepseek-r1"
]

# Standard test prompts
DEFAULT_PROMPTS = [
    "Explain the concept of artificial intelligence in simple terms.",
    "Write a short poem about technology."
]

class ExoBenchmark:
    def __init__(self, models=None, prompts=None, runs=3, timeout=300):
        """Initialize the benchmarking class with models, prompts, and settings."""
        self.models = models if models else DEFAULT_MODELS
        self.prompts = prompts if prompts else DEFAULT_PROMPTS
        self.runs = runs
        self.timeout = timeout
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def benchmark_model(self, model, prompt, run_number):
        """Benchmark a single model with a single prompt."""
        print(f"\n===== Testing {model} (Run {run_number}/{self.runs}) =====")
        print(f"Prompt: {prompt[:50]}...")
        
        # Start timing
        start_time = time.time()
        
        # Run exo with the model and prompt
        try:
            # Run with timeout to prevent hanging
            process = subprocess.run(
                ["exo", "run", model, "--prompt", prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            # Calculate metrics
            end_time = time.time()
            latency = end_time - start_time
            output = process.stdout
            output_length = len(output)
            
            print(f"Latency: {latency:.2f} seconds")
            print(f"Output length: {output_length} characters")
            print(f"First 100 chars of output: {output[:100]}...")
            
            # Store result
            result = {
                'model': model,
                'prompt': prompt,
                'run': run_number,
                'latency': latency,
                'output_length': output_length,
                'success': True
            }
            
            return result
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: Model {model} took too long to respond")
            result = {
                'model': model,
                'prompt': prompt,
                'run': run_number,
                'latency': self.timeout,
                'output_length': 0,
                'success': False
            }
            return result
        except Exception as e:
            print(f"ERROR: {str(e)}")
            result = {
                'model': model,
                'prompt': prompt,
                'run': run_number,
                'latency': 0,
                'output_length': 0,
                'success': False
            }
            return result
    
    def run_benchmarks(self):
        """Run benchmarks for all models and prompts."""
        for model in self.models:
            for prompt in self.prompts:
                for run in range(1, self.runs + 1):
                    result = self.benchmark_model(model, prompt, run)
                    self.results.append(result)
        
        return self.results
    
    def save_results(self, filename=None):
        """Save benchmark results to a CSV file."""
        if not filename:
            filename = f"benchmark_results_{self.timestamp}.csv"
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
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
            "latency": ["mean", "min", "max", "std"],
            "output_length": "mean"
        })
        
        # Flatten the multi-index columns
        model_metrics.columns = ['_'.join(col).strip() for col in model_metrics.columns.values]
        model_metrics = model_metrics.reset_index()
        
        # Round values for better display
        for col in model_metrics.columns:
            if col != "model":
                model_metrics[col] = model_metrics[col].round(2)
        
        # Create a formatted table
        table_data = []
        for _, row in model_metrics.iterrows():
            table_data.append({
                "Model": row["model"],
                "Avg Latency (s)": row["latency_mean"],
                "Min Latency (s)": row["latency_min"],
                "Max Latency (s)": row["latency_max"],
                "Std Dev": row["latency_std"],
                "Avg Output Length": row["output_length_mean"]
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
            f.write("\nPrompts used: " + str(len(self.prompts)))
            
            # Add prompt details
            f.write("\n\nPrompts:\n")
            for i, prompt in enumerate(self.prompts):
                f.write(f"{i+1}. {prompt}\n")
        
        print(f"Comparison table saved to {table_file}")
        return table_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark LLM models in Exo')
    parser.add_argument('--models', nargs='+', help='List of models to benchmark')
    parser.add_argument('--prompts', nargs='+', help='List of prompts to use')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per model-prompt combination')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each model run')
    parser.add_argument('--output', help='Output file for results (CSV)')
    return parser.parse_args()

def main():
    """Main function to run benchmarks."""
    args = parse_arguments()
    
    # Initialize benchmarking
    benchmark = ExoBenchmark(
        models=args.models,
        prompts=args.prompts,
        runs=args.runs,
        timeout=args.timeout
    )
    
    # Run benchmarks
    print("Starting benchmarks...")
    benchmark.run_benchmarks()
    
    # Save results
    output_file = args.output if args.output else None
    benchmark.save_results(output_file)
    
    # Generate comparison table
    benchmark.generate_comparison_table()
    
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()
