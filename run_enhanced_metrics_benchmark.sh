#!/bin/bash

# Script to run the enhanced metrics benchmarking for exo-lab
# This script installs dependencies, runs the enhanced metrics benchmark, and generates comparison tables

echo "===== Exo-Lab Enhanced Metrics Benchmarking ====="
echo "Starting benchmark process..."

# Install required dependencies
echo "Installing required dependencies..."
pip install pandas tabulate psutil

# Make the script executable
chmod +x enhanced_metrics_benchmark.py

# Run the enhanced benchmark with default settings
echo "Running enhanced metrics benchmarks for all models..."
echo "This may take some time depending on your hardware and the models being tested."
python enhanced_metrics_benchmark.py --models llama-3.2-1b mistral-7b --runs 1

# Check if results were generated
if [ -f "enhanced_benchmark_results_*.csv" ]; then
    echo "Benchmark completed successfully!"
    echo "Results saved to CSV file"
    
    # Check if comparison table was created
    if [ -f "enhanced_model_comparison_*.txt" ]; then
        echo "Comparison table created successfully!"
        cat enhanced_model_comparison_*.txt
    else
        echo "Comparison table was not created. Check for errors above."
    fi
else
    echo "Benchmark failed or did not complete. Check for errors above."
fi

echo "===== Enhanced metrics benchmarking process complete ====="
