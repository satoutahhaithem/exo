#!/bin/bash

# Script to run the simple enhanced LLM benchmarking for exo-lab
# This script installs dependencies, runs the simple enhanced benchmark, and generates comparison tables

echo "===== Exo-Lab Simple Enhanced LLM Benchmarking ====="
echo "Starting benchmark process..."

# Install required dependencies
echo "Installing required dependencies..."
pip install pandas tabulate

# Make the script executable
chmod +x simple_enhanced_benchmark.py

# Run the enhanced benchmark with a smaller test set for quick testing
echo "Running simple enhanced benchmarks for testing..."
echo "This will test with a limited set of models and prompts for quick verification."
# python simple_enhanced_benchmark.py --models llama-3.1-8b llama-3.2-1b mistral-7b qwen-1.5-7b deepseek-r1 --runs 1
python simple_enhanced_benchmark.py --models llama-3.2-1b mistral-7b --runs 1

# Check if results were generated
if [ -f "benchmark_results_*.csv" ]; then
    echo "Benchmark test completed successfully!"
    echo "Results saved to CSV file"
    
    # Check if comparison table was created
    if [ -f "model_comparison_*.txt" ]; then
        echo "Comparison table created successfully!"
        cat model_comparison_*.txt
    else
        echo "Comparison table was not created. Check for errors above."
    fi
else
    echo "Benchmark test failed or did not complete. Check for errors above."
fi

echo "===== Simple enhanced benchmarking test complete ====="
