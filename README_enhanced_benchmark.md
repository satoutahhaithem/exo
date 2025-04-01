# Enhanced Benchmarking Script for Exo LLMA Models

This document provides instructions for using the enhanced benchmarking script to evaluate the performance of different LLMA models integrated in Exo.

## Overview

The enhanced benchmarking script (`enhanced_benchmark.py`) allows you to:

1. Benchmark multiple LLMA models with standardized prompts
2. Measure key performance metrics:
   - Load time
   - Inference time
   - Total execution time
   - Memory usage
   - Tokens per second throughput
   - CPU utilization
3. Generate comprehensive comparison tables
4. Create visualizations of performance metrics
5. Save detailed results to CSV files

## Available Models

The following LLMA models are available for benchmarking:

- llama-3.1-8b
- llama-3.2-1b
- mistral-7b
- qwen-1.5-7b
- deepseek-r1

## Prerequisites

- Exo framework installed
- Python 3.6+
- Required Python packages: pandas, matplotlib, psutil, tabulate, tqdm

## Usage

### Quick Start

The easiest way to run the benchmarking is to use the provided shell script:

```bash
./run_enhanced_benchmark.sh
```

This script will:
1. Install required dependencies
2. Run the enhanced benchmark with default settings
3. Generate comparison tables and visualizations
4. Display the results

### Manual Execution

You can also run the benchmarking script directly with custom parameters:

```bash
python enhanced_benchmark.py [options]
```

### Command-line Options

The script supports the following command-line options:

- `--models`: List of models to benchmark (space-separated)
  - Example: `--models llama-3.1-8b mistral-7b`
  - Default: All available models

- `--prompts`: List of prompts to use (space-separated, use quotes)
  - Example: `--prompts "Explain AI" "Write a poem"`
  - Default: Standard set of test prompts

- `--engine`: Inference engine to use
  - Example: `--engine tinygrad`
  - Default: tinygrad

- `--runs`: Number of runs per model-prompt combination
  - Example: `--runs 5`
  - Default: 3

- `--output`: Output file for results (CSV)
  - Example: `--output my_results.csv`
  - Default: benchmark_results_[timestamp].csv

- `--no-visualize`: Skip visualization generation
  - Example: `--no-visualize`

### Examples

1. Benchmark specific models:
   ```bash
   python enhanced_benchmark.py --models llama-3.1-8b mistral-7b
   ```

2. Use custom prompts:
   ```bash
   python enhanced_benchmark.py --prompts "Explain quantum computing" "Write code for bubble sort"
   ```

3. Increase number of runs for more reliable results:
   ```bash
   python enhanced_benchmark.py --runs 5
   ```

4. Benchmark all models with all options specified:
   ```bash
   python enhanced_benchmark.py --models llama-3.1-8b llama-3.2-1b mistral-7b qwen-1.5-7b deepseek-r1 --runs 3 --engine tinygrad --output full_benchmark.csv
   ```

## Output Files

The script generates the following output files:

1. **CSV Results**: `benchmark_results_[timestamp].csv`
   - Contains detailed results for each model, prompt, and run

2. **Comparison Table**: `model_comparison_[timestamp].txt`
   - Text file with a formatted table comparing all models

3. **Visualizations**: Directory `benchmark_visualizations_[timestamp]/`
   - `latency_by_model.png`: Bar chart of average inference latency
   - `memory_by_model.png`: Bar chart of average memory usage
   - `tps_by_model.png`: Bar chart of average tokens per second
   - `load_vs_inference_time.png`: Comparison of load time vs inference time

## Interpreting Results

The comparison table provides a comprehensive view of model performance with the following metrics:

- **Load Time**: Time taken to load the model (seconds)
- **Inference Time**: Time taken for actual inference (seconds)
- **Total Time**: Combined load and inference time (seconds)
- **Memory**: Additional memory used during inference (MB)
- **Tokens/Second**: Throughput in tokens per second
- **CPU**: CPU utilization percentage

Lower values are better for Load Time, Inference Time, Total Time, Memory, and CPU.
Higher values are better for Tokens/Second.

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed:
   ```bash
   pip install pandas matplotlib psutil tabulate tqdm
   ```

2. Verify that the Exo framework is properly installed and configured

3. Check that the models are available and accessible to Exo

4. For memory errors, try benchmarking models individually or using smaller models first

## Extending the Script

The script is designed to be extensible:

- Add new models to the `MODELS_TO_TEST` list
- Add new prompts to the `TEST_PROMPTS` list
- Implement additional metrics in the `benchmark_model` method
- Create new visualization types in the `visualize_results` method
