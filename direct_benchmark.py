import time
import subprocess
import csv
import os

# Models to test
MODELS = [
    "llama-3.2-1b",  # Smaller model to start with
    "llama-3.1-8b",
    "mistral-7b"
]

# Test prompts
PROMPTS = [
    "Explain artificial intelligence in simple terms.",
    "Write a short poem about technology."
]

# Results storage
results = []

def benchmark_model(model, prompt):
    print(f"\n===== Testing {model} =====")
    print(f"Prompt: {prompt}")
    
    # Start timing
    start_time = time.time()
    
    # Run exo with the model and prompt
    try:
        # Run with timeout to prevent hanging
        process = subprocess.run(
            ["exo", "run", model, "--prompt", prompt],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        output = process.stdout
        output_length = len(output)
        
        print(f"Latency: {latency:.2f} seconds")
        print(f"Output length: {output_length} characters")
        print(f"First 100 chars of output: {output[:100]}...")
        
        # Store result
        results.append({
            'model': model,
            'prompt': prompt,
            'latency': latency,
            'output_length': output_length,
            'success': True
        })
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: Model {model} took too long to respond")
        results.append({
            'model': model,
            'prompt': prompt,
            'latency': 300,  # Timeout value
            'output_length': 0,
            'success': False
        })
    except Exception as e:
        print(f"ERROR: {str(e)}")
        results.append({
            'model': model,
            'prompt': prompt,
            'latency': 0,
            'output_length': 0,
            'success': False
        })

# Run benchmarks
for model in MODELS:
    for prompt in PROMPTS:
        benchmark_model(model, prompt)

# Save results to CSV
with open('benchmark_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['model', 'prompt', 'latency', 'output_length', 'success']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print("\n===== Benchmark Summary =====")
for model in MODELS:
    model_results = [r for r in results if r['model'] == model and r['success']]
    if model_results:
        avg_latency = sum(r['latency'] for r in model_results) / len(model_results)
        print(f"{model}: Average latency {avg_latency:.2f} seconds")
    else:
        print(f"{model}: No successful benchmarks")
