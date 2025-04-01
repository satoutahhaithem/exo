#!/usr/bin/env python3
import time
import subprocess
import pandas as pd
from tabulate import tabulate
import argparse
import os
import psutil
import re
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

# Quality evaluation prompts - specific prompts designed to test model capabilities
QUALITY_EVAL_PROMPTS = {
    "reasoning": "If a train travels at 120 km/h and needs to cover a distance of 450 km, how long will the journey take? Show your reasoning step by step.",
    "creativity": "Write a creative short story about a robot discovering emotions for the first time.",
    "knowledge": "Explain the process of photosynthesis and why it's important for life on Earth.",
    "instruction": "Write a recipe for chocolate chip cookies, including ingredients and step-by-step instructions."
}

class EnhancedBenchmark:
    def __init__(self, models=None, prompts=None, quality_eval=False, runs=3, timeout=300):
        """Initialize the benchmarking class with models, prompts, and settings."""
        self.models = models if models else DEFAULT_MODELS
        self.prompts = prompts if prompts else DEFAULT_PROMPTS
        self.quality_eval = quality_eval
        self.runs = runs
        self.timeout = timeout
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def benchmark_model(self, model, prompt, run_number):
        """Benchmark a single model with a single prompt."""
        print(f"\n===== Testing {model} (Run {run_number}/{self.runs}) =====")
        print(f"Prompt: {prompt[:50]}...")
        
        # Measure memory before
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # Start timing for total latency
        start_time = time.time()
        
        # Run exo with the model and prompt
        try:
            # Use Popen to capture output in real-time for first token timing
            process = subprocess.Popen(
                ["exo", "run", model, "--prompt", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Variables to track first token time
            first_token_time = None
            first_token_detected = False
            output = ""
            
            # Read output in real-time to detect first token
            while True:
                # Check if process has ended
                if process.poll() is not None:
                    break
                
                # Read a chunk of output
                chunk = process.stdout.readline()
                if not chunk:
                    # No more output
                    break
                
                # Add to total output
                output += chunk
                
                # If this is the first meaningful output and we haven't detected first token yet
                if not first_token_detected and len(chunk.strip()) > 0:
                    first_token_time = time.time() - start_time
                    first_token_detected = True
                
                # Check if we've exceeded timeout
                if time.time() - start_time > self.timeout:
                    process.terminate()
                    print(f"TIMEOUT: Model {model} took too long to respond")
                    raise subprocess.TimeoutExpired(cmd="exo run", timeout=self.timeout)
            
            # Get any remaining output
            remaining_output, _ = process.communicate()
            output += remaining_output
            
            # Calculate metrics
            end_time = time.time()
            total_latency = end_time - start_time
            
            # If we never detected first token, use total time
            if not first_token_detected:
                first_token_time = total_latency
            
            # Measure memory after
            mem_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_usage = mem_after - mem_before
            
            # Calculate token/character metrics
            output_length = len(output)
            word_count = len(output.split())
            
            # Estimate token count (rough approximation)
            token_count = len(output.split()) * 1.3  # Assuming 1.3 tokens per word on average
            tokens_per_second = token_count / total_latency if total_latency > 0 else 0
            
            print(f"Total latency: {total_latency:.2f} seconds")
            print(f"Time to first token: {first_token_time:.2f} seconds")
            print(f"Memory usage: {memory_usage:.2f} MB")
            print(f"Output length: {output_length} characters, {word_count} words")
            print(f"Estimated tokens: {token_count:.0f}, {tokens_per_second:.2f} tokens/sec")
            print(f"First 100 chars of output: {output[:100]}...")
            
            # Store result
            result = {
                'model': model,
                'prompt': prompt,
                'run': run_number,
                'total_latency': total_latency,
                'first_token_time': first_token_time,
                'memory_usage': memory_usage,
                'output_length': output_length,
                'word_count': word_count,
                'token_count': token_count,
                'tokens_per_second': tokens_per_second,
                'success': True
            }
            
            # For quality evaluation prompts, add prompt type
            if self.quality_eval:
                for prompt_type, eval_prompt in QUALITY_EVAL_PROMPTS.items():
                    if prompt == eval_prompt:
                        result['prompt_type'] = prompt_type
                        break
            
            return result
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: Model {model} took too long to respond")
            result = {
                'model': model,
                'prompt': prompt,
                'run': run_number,
                'total_latency': self.timeout,
                'first_token_time': self.timeout,
                'memory_usage': 0,
                'output_length': 0,
                'word_count': 0,
                'token_count': 0,
                'tokens_per_second': 0,
                'success': False
            }
            
            # For quality evaluation prompts, add prompt type
            if self.quality_eval:
                for prompt_type, eval_prompt in QUALITY_EVAL_PROMPTS.items():
                    if prompt == eval_prompt:
                        result['prompt_type'] = prompt_type
                        break
                        
            return result
        except Exception as e:
            print(f"ERROR: {str(e)}")
            result = {
                'model': model,
                'prompt': prompt,
                'run': run_number,
                'total_latency': 0,
                'first_token_time': 0,
                'memory_usage': 0,
                'output_length': 0,
                'word_count': 0,
                'token_count': 0,
                'tokens_per_second': 0,
                'success': False
            }
            
            # For quality evaluation prompts, add prompt type
            if self.quality_eval:
                for prompt_type, eval_prompt in QUALITY_EVAL_PROMPTS.items():
                    if prompt == eval_prompt:
                        result['prompt_type'] = prompt_type
                        break
                        
            return result
    
    def run_benchmarks(self):
        """Run benchmarks for all models and prompts."""
        # Run standard prompts
        for model in self.models:
            for prompt in self.prompts:
                for run in range(1, self.runs + 1):
                    result = self.benchmark_model(model, prompt, run)
                    self.results.append(result)
        
        # Run quality evaluation prompts if enabled
        if self.quality_eval:
            print("\n===== Running Quality Evaluation Prompts =====")
            for model in self.models:
                for prompt_type, prompt in QUALITY_EVAL_PROMPTS.items():
                    print(f"\nTesting {model} on {prompt_type} prompt")
                    result = self.benchmark_model(model, prompt, 1)  # Just one run for quality eval
                    self.results.append(result)
        
        return self.results
    
    def save_results(self, filename=None):
        """Save benchmark results to a CSV file."""
        if not filename:
            filename = f"enhanced_benchmark_results_{self.timestamp}.csv"
        
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
            "total_latency": "mean",
            "first_token_time": "mean",
            "memory_usage": "mean",
            "output_length": "mean",
            "token_count": "mean",
            "tokens_per_second": "mean"
        })
        
        # Reset index to make model a column
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
                "Total Latency (s)": row["total_latency"],
                "First Token (s)": row["first_token_time"],
                "Memory (MB)": row["memory_usage"],
                "Avg Output Len": row["output_length"],
                "Avg Tokens": row["token_count"],
                "Tokens/Second": row["tokens_per_second"]
            })
        
        # Create a formatted table string
        table_str = tabulate(table_data, headers="keys", tablefmt="grid")
        print("\nModel Performance Comparison:")
        print(table_str)
        
        # Save the table to a file
        table_file = f"enhanced_model_comparison_{self.timestamp}.txt"
        with open(table_file, "w") as f:
            f.write("Enhanced Model Performance Comparison\n")
            f.write("===================================\n\n")
            f.write(table_str)
            f.write("\n\nBenchmark conducted on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write("\nNumber of runs per model-prompt combination: " + str(self.runs))
            f.write("\nPrompts used: " + str(len(self.prompts)))
            
            # Add prompt details
            f.write("\n\nStandard Prompts:\n")
            for i, prompt in enumerate(self.prompts):
                f.write(f"{i+1}. {prompt}\n")
            
            # Add quality evaluation details if used
            if self.quality_eval:
                f.write("\nQuality Evaluation Prompts:\n")
                for prompt_type, prompt in QUALITY_EVAL_PROMPTS.items():
                    f.write(f"- {prompt_type}: {prompt}\n")
        
        print(f"Comparison table saved to {table_file}")
        return table_file
    
    def generate_quality_evaluation_table(self):
        """Generate a table comparing model performance on quality evaluation prompts."""
        if not self.quality_eval or not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        
        # Filter for quality evaluation prompts and successful runs
        df_quality = df[df["success"] == True]
        df_quality = df_quality[df_quality["prompt"].isin(QUALITY_EVAL_PROMPTS.values())]
        
        if df_quality.empty:
            print("No successful quality evaluation runs to analyze.")
            return None
        
        # Group by model and prompt type
        quality_metrics = df_quality.groupby(["model", "prompt_type"]).agg({
            "total_latency": "mean",
            "output_length": "mean",
            "token_count": "mean"
        }).reset_index()
        
        # Round values
        for col in quality_metrics.columns:
            if col not in ["model", "prompt_type"]:
                quality_metrics[col] = quality_metrics[col].round(2)
        
        # Create a pivot table for better visualization
        pivot_table = quality_metrics.pivot(index="model", columns="prompt_type", values=["total_latency", "output_length", "token_count"])
        
        # Format and save the table
        quality_table_file = f"quality_evaluation_{self.timestamp}.txt"
        with open(quality_table_file, "w") as f:
            f.write("Quality Evaluation Results by Task Type\n")
            f.write("=====================================\n\n")
            
            # Write latency comparison
            f.write("Latency by Task Type (seconds)\n")
            latency_table = quality_metrics.pivot(index="model", columns="prompt_type", values="total_latency")
            f.write(tabulate(latency_table, headers="keys", tablefmt="grid"))
            f.write("\n\n")
            
            # Write output length comparison
            f.write("Output Length by Task Type (characters)\n")
            length_table = quality_metrics.pivot(index="model", columns="prompt_type", values="output_length")
            f.write(tabulate(length_table, headers="keys", tablefmt="grid"))
            f.write("\n\n")
            
            # Write token count comparison
            f.write("Token Count by Task Type\n")
            token_table = quality_metrics.pivot(index="model", columns="prompt_type", values="token_count")
            f.write(tabulate(token_table, headers="keys", tablefmt="grid"))
            
            f.write("\n\nQuality Evaluation conducted on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        print(f"Quality evaluation results saved to {quality_table_file}")
        return quality_table_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Benchmark for LLM models in Exo')
    parser.add_argument('--models', nargs='+', help='List of models to benchmark')
    parser.add_argument('--prompts', nargs='+', help='List of prompts to use')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per model-prompt combination')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each model run')
    parser.add_argument('--output', help='Output file for results (CSV)')
    parser.add_argument('--quality-eval', action='store_true', help='Run additional quality evaluation prompts')
    return parser.parse_args()

def main():
    """Main function to run benchmarks."""
    args = parse_arguments()
    
    # Initialize benchmarking
    benchmark = EnhancedBenchmark(
        models=args.models,
        prompts=args.prompts,
        quality_eval=args.quality_eval,
        runs=args.runs,
        timeout=args.timeout
    )
    
    # Run benchmarks
    print("Starting enhanced benchmarks...")
    benchmark.run_benchmarks()
    
    # Save results
    output_file = args.output if args.output else None
    benchmark.save_results(output_file)
    
    # Generate comparison table
    benchmark.generate_comparison_table()
    
    # Generate quality evaluation table if quality eval was run
    if args.quality_eval:
        benchmark.generate_quality_evaluation_table()
    
    print("Enhanced benchmarking complete!")

if __name__ == "__main__":
    main()
