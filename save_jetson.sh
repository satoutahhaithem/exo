#!/bin/bash
# Usage: ./run_jetson_benchmark.sh

# Make sure the "jetson" directory exists
mkdir -p jetson

# Run the Python script and direct all output to jetson/benchmark
python3 enhanced_metrics_benchmark.py > jetson/benchmark

# Optional: Show a message indicating where the output was saved
echo "Output saved to jetson/benchmark"
