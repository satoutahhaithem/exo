# Exo-Lab LLM Benchmarking

This project provides a framework for benchmarking different LLM models within the same node using the exo-lab clustering service.

## Project Structure

- `enhanced_metrics_benchmark.py` - Enhanced benchmark script with additional performance metrics
- `models_for_testing.md` - Documentation of selected models for benchmarking
- `benchmarking_methodology.md` - Detailed explanation of benchmarking methodology and metrics

## Getting Started

### Prerequisites

- Python 3.12+ (required by exo)
- For NVIDIA GPU support on Linux:
  - NVIDIA driver (verify with `nvidia-smi`)
  - CUDA toolkit (verify with `nvcc --version`)
  - cuDNN library

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/exo-benchmark.git
   cd exo-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install pandas matplotlib psutil
   ```

3. Make the benchmark script executable:
   ```bash
   chmod +x run_enhanced_metrics_benchmark.sh
   ```

### Running Benchmarks

Execute the enhanced benchmark script:
```bash
python enhanced_metrics_benchmark.py
```

This will:
1. Install required dependencies
2. Run benchmarks for all models
3. Generate visualizations of the results

## Benchmarking Details

### Models Tested

- Llama 3.1 8B (`llama-3.1-8b`)
- Llama 3.2 1B (`llama-3.2-1b`)
- Mistral 7B (`mistral-7b`)
- Qwen 1.5 7B (`qwen-1.5-7b`)
- DeepSeek R1 (`deepseek-r1`)

### Metrics Measured

- Inference Latency: Time from prompt submission to complete response
- Memory Usage: Additional memory consumed during model inference
- Tokens Per Second: Rate at which the model generates output tokens

### Test Prompts

The benchmarks use standardized prompts covering different tasks:
1. Explanation of artificial intelligence
2. Creative writing (poem about technology)
3. Summarization (benefits of renewable energy)
4. Technical comparison (classical vs. quantum computing)
5. Scientific explanation (photosynthesis process)

## Results

After running the benchmarks, results will be saved to:
- `benchmark_results.csv` - Raw benchmark data
- `latency_by_model.png` - Visualization of average latency by model
- `memory_by_model.png` - Visualization of average memory usage by model
- `tps_by_model.png` - Visualization of average tokens per second by model

## Customization

You can modify the benchmark script to:
- Test different models by editing the `MODELS_TO_TEST` list
- Use different prompts by editing the `TEST_PROMPTS` list
- Change the inference engine from `tinygrad` to `mlx` for Apple Silicon devices

## Limitations

- Results are specific to the hardware and software configuration used
- Performance may vary with different prompts or workloads
- Quality of model outputs is not evaluated, only performance metrics
- The methodology focuses on single-node performance, not distributed inference

## Enhanced Metrics Benchmarking for Exo LLMA Models

This document provides instructions for using the enhanced metrics benchmarking script to evaluate the performance of different LLMA models in Exo with additional metrics as requested.

### New Metrics Implemented

- **Time to First Token**: Measures how quickly the model begins generating output
- **Token/Character Usage**: Counts output length in characters, words, and estimated tokens
- **Tokens Per Second**: Calculates throughput rate
- **Quality Evaluation**: Tests models on specific task types (reasoning, creativity, knowledge, instruction)

Standard deviation has been removed from the metrics as requested.

### Available Models

- llama-3.1-8b
- llama-3.2-1b

### Prerequisites

- Exo framework installed
- Python 3.6+
- Required packages: pandas, tabulate, psutil

### Usage

#### Quick Start

```bash
./run_enhanced_metrics_benchmark.sh
```

This script will:
1. Install dependencies
2. Run the enhanced benchmark with default settings
3. Generate comparison tables with the new metrics
4. Display the results

#### Manual Execution

```bash
python enhanced_metrics_benchmark.py [options]
```

#### Command-line Options

- --models: List of models to benchmark (e.g., `--models llama-3.1-8b`)
- --prompts: List of prompts (e.g., `--prompts "Explain AI" "Write a poem"`)
- --runs: Number of runs per model-prompt combination (default: 3)
- --timeout: Timeout in seconds (default: 300)
- --output: Results CSV file (default: enhanced_benchmark_results_[timestamp].csv)
- --quality-eval: Run additional quality evaluation prompts

#### Examples

1. Benchmark specific models:
   ```bash
   python enhanced_metrics_benchmark.py --models llama-3.1-8b
   ```

2. Use custom prompts:
   ```bash
   python enhanced_metrics_benchmark.py --prompts "Explain quantum computing" "Write code for bubble sort"
   ```

3. Run with quality evaluation:
   ```bash
   python enhanced_metrics_benchmark.py --quality-eval
   ```

4. Increase timeout:
   ```bash
   python enhanced_metrics_benchmark.py --timeout 600
   ```

### Output Files

- CSV Results: `enhanced_benchmark_results_[timestamp].csv`
- Comparison Table: `enhanced_model_comparison_[timestamp].txt`
- Quality Evaluation (if enabled): `quality_evaluation_[timestamp].txt`

### Interpreting Results

- **Total Latency**: Total time from prompt submission to complete response (seconds)
- **First Token**: Time until the first output token (seconds)
- **Memory**: Additional memory used during inference (MB)
- **Avg Output Len**: Average length (characters)
- **Avg Tokens**: Estimated tokens in the output
- **Tokens/Second**: Throughput in tokens per second

Lower values are better for Total Latency, First Token, and Memory.
Higher values are better for Tokens/Second.

### Troubleshooting

1. Install dependencies:
   ```bash
   pip install pandas tabulate psutil
   ```
2. Ensure the Exo framework is installed.
3. Verify model availability.
4. For timeout issues, try:
   ```bash
   python enhanced_metrics_benchmark.py --timeout 600
   ```
5. For memory issues, adjust permissions or tracking methods.

### Extending the Script

- Add new models to the `DEFAULT_MODELS` list.
- Add prompts to the `DEFAULT_PROMPTS` list.
- Add quality evaluation prompts to the `QUALITY_EVAL_PROMPTS` dictionary.
- Enhance metrics in the `benchmark_model` method.
