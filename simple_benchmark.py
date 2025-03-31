import time
import os
import subprocess

def benchmark_model(model_name, prompt):
    print(f"Testing model: {model_name}")
    print(f"Prompt: {prompt[:30]}...")
    
    start_time = time.time()
    result = subprocess.run(
        ["exo", "run", model_name, "--prompt", prompt],
        capture_output=True,
        text=True
    )
    end_time = time.time()
    
    print(f"Latency: {end_time - start_time:.2f} seconds")
    print("-" * 40)
    return end_time - start_time

# Test different models
models = ["llama-3.1-8b", "llama-3.2-1b", "mistral-7b"]
prompt = "Explain the concept of artificial intelligence in simple terms."

for model in models:
    benchmark_model(model, prompt)
