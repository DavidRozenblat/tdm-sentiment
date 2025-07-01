"""
Model Comparison Script: Original FinBERT vs. Optimized FinBERT

This script evaluates and compares the performance of two models:
- Original FinBERT model (finbert_local)
- Optimized FinBERT model (optimized_finbert)

Both models are tested on the same dataset (Newyork20042023) to measure:
- Execution time
- Output similarity
- Memory usage
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from tabulate import tabulate
import json

SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
import sys
sys.path.append(str(SRC_PATH))
from config import *

# Dataset and model paths
CORPUS_NAME = "Newyork20042023"
RESULT_FOLDER_PATH = ''
ORIGINAL_MODEL_PATH = FINBERT_LOCAL_MODEL_PATH
OPTIMIZED_MODEL_PATH = OPTIMIZED_FINBERT_MODEL_PATH

# Number of samples for comparison (0 for all samples)
NUM_SAMPLES = 100

def title_sentiment_probs(text, sentiment_analyzer):
    """
    Get the sentiment probabilities of a single title text.
    """
    # Note: The function expects a list of texts; we wrap text in a list.
    try:
        return sentiment_analyzer.get_sentiment_dict(text)
    except Exception as e:
        print(f'error: {e}, input was:{text}')
        return None


def load_models():
    """
    Load both the original and optimized FinBERT models.
    Returns:
        tuple: (original_model, optimized_model)
    """
    print("\nLoading models...")
    
    # Load original FinBERT model
    start_time = time.time()
    original_model = sentiment_score.TextAnalysis(ORIGINAL_MODEL_PATH)
    original_load_time = time.time() - start_time
    print(f"Original model loaded in {original_load_time:.2f}s")
    
    # Load optimized FinBERT model
    start_time = time.time()
    optimized_model = optimized_model1.OptimizedFinBERT(
        local_model_path=OPTIMIZED_MODEL_PATH,
        original_model_path=ORIGINAL_MODEL_PATH,
        quantization_type="int8",
        force_optimization=False
    )
    optimized_load_time = time.time() - start_time
    print(f"Optimized model loaded in {optimized_load_time:.2f}s")
    
    return original_model, optimized_model, {"original": original_load_time, "optimized": optimized_load_time}

def process_samples(model, samples):
    """
    Process a list of samples with the given model.
    
    Args:
        model: The sentiment model to use
        samples: List of text samples to process
        
    Returns:
        tuple: (results, execution_time)
    """
    results = []
    start_time = time.time()
    
    for text in tqdm(samples, desc=f"Processing samples"):
        sentiment_dict = title_sentiment_probs(text, model)
        results.append(sentiment_dict)
    
    execution_time = time.time() - start_time
    return results, execution_time

def calculate_memory_usage(model):
    """
    Calculate approximate memory usage of a model.
    
    Args:
        model: The model to measure
        
    Returns:
        float: Memory usage in MB
    """
    memory_mb = 0
    try:
        # Get model size by summing parameter sizes
        total_params = sum(p.numel() for p in model.sentiment_pipeline.model.parameters())
        # Multiply by 4 for standard float32 size (4 bytes)
        bytes_per_param = 4
        if hasattr(model, 'quantization_type') and model.quantization_type == 'int8':
            bytes_per_param = 1  # 1 byte per parameter for int8
        memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
    except Exception as e:
        print(f"Error measuring memory: {e}")
        memory_mb = "N/A"
    
    return memory_mb

def compare_results(original_results, optimized_results):
    """
    Compare the results from both models to check for differences.
    
    Args:
        original_results: Results from the original model
        optimized_results: Results from the optimized model
        
    Returns:
        dict: Statistics about the differences
    """
    diff_stats = {
        "total_samples": len(original_results),
        "identical_outputs": 0,
        "mean_diff": {"positive": 0, "negative": 0, "neutral": 0},
        "max_diff": {"positive": 0, "negative": 0, "neutral": 0},
        "identical_classifications": 0
    }
    
    diffs = []
    
    for i, (orig, opt) in enumerate(zip(original_results, optimized_results)):
        if orig is None or opt is None:
            continue
            
        # Check if results are identical
        identical = all(abs(orig.get(k, 0) - opt.get(k, 0)) < 1e-5 for k in set(orig) | set(opt))
        if identical:
            diff_stats["identical_outputs"] += 1
            
        # Calculate differences for each sentiment class
        sample_diffs = {}
        for label in ["positive", "negative", "neutral"]:
            orig_val = orig.get(label, 0)
            opt_val = opt.get(label, 0)
            diff = abs(orig_val - opt_val)
            sample_diffs[label] = diff
            
            # Update max difference
            if diff > diff_stats["max_diff"][label]:
                diff_stats["max_diff"][label] = diff
                
        diffs.append(sample_diffs)
        
        # Check if classification is the same (highest probability class)
        orig_class = max(orig.items(), key=lambda x: x[1])[0] if orig else None
        opt_class = max(opt.items(), key=lambda x: x[1])[0] if opt else None
        if orig_class == opt_class:
            diff_stats["identical_classifications"] += 1
    
    # Calculate mean differences
    for label in ["positive", "negative", "neutral"]:
        values = [d[label] for d in diffs if label in d]
        if values:
            diff_stats["mean_diff"][label] = sum(values) / len(values)
    
    return diff_stats

def format_comparison_table(execution_times, memory_usage, diff_stats, load_times):
    """
    Format comparison results as a table.
    
    Args:
        execution_times: Dictionary with execution times
        memory_usage: Dictionary with memory usage
        diff_stats: Statistics about output differences
        load_times: Dictionary with model load times
        
    Returns:
        str: Formatted table
    """
    # Create a pandas DataFrame for better presentation
    data = {
        "Metric": [
            "Load Time (s)",
            "Execution Time (s)",
            "Memory Usage (MB)",
            "Classification Agreement (%)",
            "Mean Diff - Positive",
            "Mean Diff - Negative",
            "Mean Diff - Neutral",
            "Max Diff - Positive",
            "Max Diff - Negative", 
            "Max Diff - Neutral"
        ],
        "Original FinBERT": [
            f"{load_times['original']:.2f}",
            f"{execution_times['original']:.2f}",
            f"{memory_usage['original']:.2f}" if isinstance(memory_usage['original'], (int, float)) else memory_usage['original'],
            "100.0", # Base reference
            "0.0",    # Base reference
            "0.0",    # Base reference
            "0.0",    # Base reference
            "0.0",    # Base reference
            "0.0",    # Base reference
            "0.0"     # Base reference
        ],
        "Optimized FinBERT": [
            f"{load_times['optimized']:.2f}",
            f"{execution_times['optimized']:.2f}",
            f"{memory_usage['optimized']:.2f}" if isinstance(memory_usage['optimized'], (int, float)) else memory_usage['optimized'],
            f"{(diff_stats['identical_classifications'] / diff_stats['total_samples']) * 100:.2f}",
            f"{diff_stats['mean_diff']['positive']:.6f}",
            f"{diff_stats['mean_diff']['negative']:.6f}",
            f"{diff_stats['mean_diff']['neutral']:.6f}",
            f"{diff_stats['max_diff']['positive']:.6f}",
            f"{diff_stats['max_diff']['negative']:.6f}",
            f"{diff_stats['max_diff']['neutral']:.6f}"
        ],
        "Comparison": [
            f"{load_times['original'] / load_times['optimized']:.2f}x faster" if load_times['optimized'] < load_times['original'] else 
            f"{load_times['optimized'] / load_times['original']:.2f}x slower",
            
            f"{execution_times['original'] / execution_times['optimized']:.2f}x faster" if execution_times['optimized'] < execution_times['original'] else 
            f"{execution_times['optimized'] / execution_times['original']:.2f}x slower",
            
            "Smaller" if (isinstance(memory_usage['optimized'], (int, float)) and 
                          isinstance(memory_usage['original'], (int, float)) and 
                          memory_usage['optimized'] < memory_usage['original']) else "Larger",
            
            f"{(diff_stats['identical_classifications'] / diff_stats['total_samples']) * 100:.2f}% match",
            f"±{diff_stats['mean_diff']['positive']:.6f}",
            f"±{diff_stats['mean_diff']['negative']:.6f}",
            f"±{diff_stats['mean_diff']['neutral']:.6f}",
            f"±{diff_stats['max_diff']['positive']:.6f}",
            f"±{diff_stats['max_diff']['negative']:.6f}",
            f"±{diff_stats['max_diff']['neutral']:.6f}"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Format as string table
    return tabulate(df, headers='keys', tablefmt='pipe', showindex=False)

def main():
    """
    Main function to run the model comparison.
    """
    print(f"===== Comparing Original FinBERT vs. Optimized FinBERT =====")
    print(f"Dataset: {CORPUS_NAME}")
    
    # Load the dataset
    try:
        file_path = list(RESULT_FOLDER_PATH.glob("chunk_*.csv"))[0]
        print(f"\nLoading dataset from {file_path}")
        df = pd.read_csv(file_path)
        
        # Select a sample of rows
        if NUM_SAMPLES > 0 and NUM_SAMPLES < len(df):
            df = df.sample(NUM_SAMPLES, random_state=42)
            
        # Get titles for sentiment analysis
        samples = df['title'].tolist()
        print(f"Loaded {len(samples)} samples for comparison")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # If we can't load the dataset, use some example texts
        samples = [
            "Stock market reaches new high amid economic recovery",
            "Federal Reserve announces interest rate hike",
            "Company profits plunge in fourth quarter",
            "Unemployment rate falls to 4.5 percent",
            "Tech stocks tumble on regulatory concerns",
            "Oil prices surge due to supply constraints",
            "Bank announces higher than expected dividends",
            "Inflation reaches 40-year high, concerning economists",
            "Real estate market shows signs of cooling",
            "New trade agreement expected to boost exports"
        ]
        print(f"Using {len(samples)} example samples instead")
    
    # Load models
    original_model, optimized_model, load_times = load_models()
    
    # Measure memory usage
    print("\nMeasuring memory usage...")
    memory_usage = {
        "original": calculate_memory_usage(original_model),
        "optimized": calculate_memory_usage(optimized_model)
    }
    
    # Process samples with original model
    print("\nProcessing with original FinBERT model...")
    original_results, original_time = process_samples(original_model, samples)
    
    # Process samples with optimized model
    print("\nProcessing with optimized FinBERT model...")
    optimized_results, optimized_time = process_samples(optimized_model, samples)
    
    # Compare results
    print("\nComparing results...")
    diff_stats = compare_results(original_results, optimized_results)
    
    # Format execution times
    execution_times = {
        "original": original_time,
        "optimized": optimized_time
    }
    
    # Display summary table
    print("\n===== COMPARISON RESULTS =====\n")
    comparison_table = format_comparison_table(execution_times, memory_usage, diff_stats, load_times)
    print(comparison_table)
    
    # Save results to file
    results = {
        "dataset": CORPUS_NAME,
        "num_samples": len(samples),
        "load_times": load_times,
        "execution_times": execution_times,
        "memory_usage": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in memory_usage.items()},
        "diff_stats": diff_stats
    }
    
    with open("model_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to model_comparison_results.json")
    
    print("\n===== CONCLUSION =====")
    speedup = original_time / optimized_time if optimized_time > 0 else 0
    accuracy = (diff_stats['identical_classifications'] / diff_stats['total_samples']) * 100
    
    if speedup > 1.5 and accuracy > 95:
        print("The optimized model is significantly faster while maintaining high accuracy.")
        print("Recommendation: Use the optimized model for production.")
    elif speedup > 1.5 and accuracy > 90:
        print("The optimized model is significantly faster with acceptable accuracy.")
        print("Recommendation: Use the optimized model for most applications, but verify outputs for critical applications.")
    elif speedup > 1.2 and accuracy > 98:
        print("The optimized model is somewhat faster with near-identical results.")
        print("Recommendation: Use the optimized model for all applications.")
    else:
        print(f"The optimized model is {speedup:.2f}x faster with {accuracy:.2f}% classification agreement.")
        print("Recommendation: Choose based on your specific requirements for speed vs. precision.")

if __name__ == "__main__":
    main()