#!/bin/bash

# Script to run the LLM benchmarking for exo-lab
# This script installs dependencies, runs the benchmark, and generates visualizations

echo "===== Exo-Lab LLM Benchmarking ====="
echo "Starting benchmark process..."

# Install required dependencies
echo "Installing required dependencies..."
pip install pandas matplotlib psutil

# Run the benchmark
echo "Running benchmarks for all models..."
echo "This may take some time depending on your hardware and the models being tested."
python benchmark.py

# Check if results were generated
if [ -f "benchmark_results.csv" ]; then
    echo "Benchmark completed successfully!"
    echo "Results saved to benchmark_results.csv"
    
    # Check if visualizations were created
    if [ -f "latency_by_model.png" ]; then
        echo "Visualizations created successfully!"
    else
        echo "Visualizations were not created. You may need to install matplotlib and pandas."
    fi
else
    echo "Benchmark failed or did not complete. Check for errors above."
fi

echo "===== Benchmarking process complete ====="
