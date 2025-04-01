#!/bin/bash

# Script to run the enhanced LLM benchmarking for exo-lab
# This script installs dependencies, runs the enhanced benchmark, and generates visualizations and comparison tables

echo "===== Exo-Lab Enhanced LLM Benchmarking ====="
echo "Starting benchmark process..."

# Install required dependencies
echo "Installing required dependencies..."
pip install pandas matplotlib psutil tabulate tqdm

# Make the script executable
chmod +x enhanced_benchmark.py

# Run the enhanced benchmark
echo "Running enhanced benchmarks for all models..."
echo "This may take some time depending on your hardware and the models being tested."
python enhanced_benchmark.py

# Check if results were generated
if [ -f "benchmark_results_*.csv" ]; then
    echo "Benchmark completed successfully!"
    echo "Results saved to CSV file"
    
    # Check if comparison table was created
    if [ -f "model_comparison_*.txt" ]; then
        echo "Comparison table created successfully!"
        cat model_comparison_*.txt
    else
        echo "Comparison table was not created. Check for errors above."
    fi
    
    # Check if visualizations directory exists
    if [ -d "benchmark_visualizations_*" ]; then
        echo "Visualizations created successfully!"
    else
        echo "Visualizations were not created. Check for errors above."
    fi
else
    echo "Benchmark failed or did not complete. Check for errors above."
fi

echo "===== Enhanced benchmarking process complete ====="
