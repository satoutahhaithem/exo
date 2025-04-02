# LLM Models for Benchmarking in Exo-Lab

This document identifies the LLM models selected for benchmarking within the same node using the exo-lab clustering service.

## Selected Models

Based on the exo repository and its supported models, we've selected the following models for benchmarking:

1. Llama 3.1 8B (`llama-3.1-8b`)
   - Size: 8 billion parameters
   - Architecture: Transformer-based
   - Developer: Meta AI
   - Use case: General purpose text generation and understanding

2. Llama 3.2 1B (`llama-3.2-1b`)
   - Size: 1 billion parameters
   - Architecture: Transformer-based
   - Developer: Meta AI
   - Use case: Lightweight general purpose text generation

3. Mistral 7B (`mistral-7b`)
   - Size: 7 billion parameters
   - Architecture: Transformer-based with Sliding Window Attention
   - Developer: Mistral AI
   - Use case: General purpose text generation with strong reasoning capabilities

4. Qwen 1.5 7B (`qwen-1.5-7b`)
   - Size: 7 billion parameters
   - Architecture: Transformer-based
   - Developer: Alibaba Cloud
   - Use case: Multilingual text generation and understanding

5. DeepSeek R1 (`deepseek-r1`)
   - Size: Varies (typically 7B)
   - Architecture: Transformer-based
   - Developer: DeepSeek
   - Use case: Code generation and technical reasoning

## Selection Criteria

These models were selected based on the following criteria:

1. Diversity of architectures: The selection includes models from different developers with varying architectural approaches.
2. Range of model sizes: From 1B to 8B parameters, allowing comparison of performance across different model sizes.
3. Compatibility with exo: All selected models are officially supported by the exo framework.
4. Popularity and relevance: These models represent some of the most widely used open-source LLMs.
5. Ability to run on a single node: All models can be run on a single node with sufficient resources.

## Benchmarking Approach

Each model will be benchmarked using the same set of standardized prompts to ensure fair comparison. The benchmarking will measure:
- Inference latency
- Memory usage
- Tokens per second throughput

Results will be collected, analyzed, and visualized to provide insights into the relative performance of these models when running on the same node within the exo-lab clustering service.
