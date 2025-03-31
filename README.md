# Exo-Lab LLM Benchmarking

This project provides a framework for benchmarking different LLM models within the same node using the exo-lab clustering service.

## Project Structure

- `benchmark.py` - Main Python script for running benchmarks
- `run_benchmark.sh` - Shell script to automate the benchmarking process
- `models_for_testing.md` - Documentation of selected models for benchmarking
- `benchmarking_methodology.md` - Detailed explanation of benchmarking methodology and metrics
- `todo.md` - Project task list and progress tracking

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
   chmod +x run_benchmark.sh
   ```

### Running Benchmarks

Execute the benchmark script:
```bash
./run_benchmark.sh
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

- **Inference Latency**: Time from prompt submission to complete response
- **Memory Usage**: Additional memory consumed during model inference
- **Tokens Per Second**: Rate at which the model generates output tokens

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
