# Enhanced Metrics Benchmarking for Exo LLMA Models

This document provides instructions for using the enhanced metrics benchmarking script to evaluate the performance of different LLMA models in Exo with additional metrics as requested.

## New Metrics Implemented

The enhanced benchmarking script now includes the following metrics:

1. **Memory Usage**: Tracks RAM consumption during model execution
2. **Time to First Token**: Measures how quickly the model begins generating output
3. **Token/Character Usage**: Counts output length in characters, words, and estimated tokens
4. **Tokens Per Second**: Calculates throughput rate
5. **Quality Evaluation**: Tests models on specific task types (reasoning, creativity, knowledge, instruction)

Standard deviation has been removed from the metrics as requested.

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
- Required Python packages: pandas, tabulate, psutil

## Usage

### Quick Start

The easiest way to run the enhanced benchmarking is to use the provided shell script:

```bash
./run_enhanced_metrics_benchmark.sh
```

This script will:
1. Install required dependencies
2. Run the enhanced benchmark with default settings
3. Generate comparison tables with the new metrics
4. Display the results

### Manual Execution

You can also run the benchmarking script directly with custom parameters:

```bash
python enhanced_metrics_benchmark.py [options]
```

### Command-line Options

The script supports the following command-line options:

- `--models`: List of models to benchmark (space-separated)
  - Example: `--models llama-3.1-8b mistral-7b`
  - Default: All available models

- `--prompts`: List of prompts to use (space-separated, use quotes)
  - Example: `--prompts "Explain AI" "Write a poem"`
  - Default: Standard set of test prompts

- `--runs`: Number of runs per model-prompt combination
  - Example: `--runs 5`
  - Default: 3

- `--timeout`: Timeout in seconds for each model run
  - Example: `--timeout 600`
  - Default: 300 (5 minutes)

- `--output`: Output file for results (CSV)
  - Example: `--output my_results.csv`
  - Default: enhanced_benchmark_results_[timestamp].csv

- `--quality-eval`: Run additional quality evaluation prompts
  - Example: `--quality-eval`
  - Default: Not enabled

### Examples

1. Benchmark specific models:
   ```bash
   python enhanced_metrics_benchmark.py --models llama-3.1-8b mistral-7b
   ```

2. Use custom prompts:
   ```bash
   python enhanced_metrics_benchmark.py --prompts "Explain quantum computing" "Write code for bubble sort"
   ```

3. Run with quality evaluation:
   ```bash
   python enhanced_metrics_benchmark.py --quality-eval
   ```

4. Increase timeout for larger models:
   ```bash
   python enhanced_metrics_benchmark.py --timeout 600
   ```

## Output Files

The script generates the following output files:

1. **CSV Results**: `enhanced_benchmark_results_[timestamp].csv`
   - Contains detailed results for each model, prompt, and run with all metrics

2. **Comparison Table**: `enhanced_model_comparison_[timestamp].txt`
   - Text file with a formatted table comparing all models on the new metrics

3. **Quality Evaluation** (if enabled): `quality_evaluation_[timestamp].txt`
   - Comparison of models on different task types

## Interpreting Results

The comparison table provides a comprehensive view of model performance with the following metrics:

- **Total Latency**: Total time from prompt submission to complete response (seconds)
- **First Token**: Time until the model produces its first output token (seconds)
- **Memory**: Additional memory used during inference (MB)
- **Avg Output Len**: Average length of model responses in characters
- **Avg Tokens**: Estimated number of tokens in the output
- **Tokens/Second**: Throughput in tokens per second

Lower values are better for Total Latency, First Token, and Memory.
Higher values are better for Tokens/Second.

## Troubleshooting

If you encounter issues:

1. Ensure all dependencies are installed:
   ```bash
   pip install pandas tabulate psutil
   ```

2. Verify that the Exo framework is properly installed and configured

3. Check that the models are available and accessible to Exo

4. For timeout errors, increase the timeout value:
   ```bash
   python enhanced_metrics_benchmark.py --timeout 600
   ```

5. If memory tracking causes issues, you may need to run with elevated permissions or modify the script to use a different memory tracking method

## Extending the Script

The script is designed to be extensible:

- Add new models to the `DEFAULT_MODELS` list
- Add new prompts to the `DEFAULT_PROMPTS` list
- Add new quality evaluation prompts to the `QUALITY_EVAL_PROMPTS` dictionary
- Implement additional metrics in the `benchmark_model` method
