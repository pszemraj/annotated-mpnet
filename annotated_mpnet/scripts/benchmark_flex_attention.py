#!/usr/bin/env python3
"""
Script to benchmark FlexAttention vs standard attention for MPNet.
This helps quantify the performance benefits of using sliding window attention.
"""

import argparse
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from rich.console import Console
from rich.table import Table

from annotated_mpnet.data import (
    DataCollatorForMaskedPermutedLanguageModeling,
    MPNetDataset,
)
from annotated_mpnet.modeling import MPNetForPretraining, MPNetFlexForPretraining


@dataclass
class Args:
    """Simple class to hold model arguments"""
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_dim: int = 3072
    encoder_attention_heads: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    max_positions: int = 512
    activation_fn: str = "gelu"
    normalize_before: bool = False
    sliding_window_size: int = None


def create_random_dataset(tokenizer, seq_length, num_samples=100):
    """Create a random dataset with specified sequence length"""
    # Create random token IDs
    vocab_size = tokenizer.vocab_size
    examples = []
    
    for _ in range(num_samples):
        # Create a random sequence of tokens
        tokens = torch.randint(100, vocab_size-100, (seq_length,))
        examples.append({"input_ids": tokens})
    
    return examples


def benchmark_model(model, dataloader, device, use_flex_attention=False):
    """Benchmark model inference speed"""
    model.eval()
    total_tokens = 0
    start_time = time.time()
    
    # Track memory usage
    start_mem = torch.cuda.memory_allocated() if device == "cuda" else 0
    max_mem = start_mem
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            device_batch = {
                data_type: (t.to(device) if isinstance(t, torch.Tensor) else t)
                for data_type, t in batch.items()
                if data_type != "attention_mask"
            }
            
            # Add number of tokens to total
            total_tokens += device_batch["ntokens"]
            
            # Run model
            if hasattr(model, "use_flex_attention"):
                # MPNetFlexForPretraining
                _ = model(**device_batch, use_flex_attention=use_flex_attention)
            else:
                # Standard MPNetForPretraining
                _ = model(**device_batch)
            
            # Track peak memory
            if device == "cuda":
                current_mem = torch.cuda.memory_allocated()
                max_mem = max(max_mem, current_mem)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate tokens per second
    tokens_per_sec = total_tokens / elapsed
    
    # Calculate memory usage in MB
    if device == "cuda":
        memory_usage = (max_mem - start_mem) / (1024 * 1024)
    else:
        memory_usage = 0
    
    return {
        "tokens_per_sec": tokens_per_sec,
        "memory_mb": memory_usage,
        "total_tokens": total_tokens,
        "time_seconds": elapsed
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlexAttention vs standard attention")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 512, 1024, 2048], 
                      help="Sequence lengths to benchmark")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[64, 128, 256, 512], 
                      help="Window sizes to benchmark")
    parser.add_argument("--num-samples", type=int, default=10, 
                      help="Number of samples to benchmark")
    parser.add_argument("--batch-size", type=int, default=1, 
                      help="Batch size for benchmarking")
    parser.add_argument("--reduced-model", action="store_true",
                      help="Use a reduced model size for testing on smaller GPUs")
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    
    # Initialize data collator
    collator = DataCollatorForMaskedPermutedLanguageModeling(tokenizer=tokenizer)
    
    # Create console for pretty printing
    console = Console()
    
    # Create model arguments
    if args.reduced_model:
        # Smaller model for testing
        model_args = Args(
            encoder_layers=4,
            encoder_embed_dim=256,
            encoder_ffn_dim=1024,
            encoder_attention_heads=4,
        )
    else:
        model_args = Args()
    
    # Create standard MPNet model
    standard_model = MPNetForPretraining(model_args, tokenizer)
    standard_model.to(device)
    
    # Results table
    table = Table(title="FlexAttention vs Standard Attention Benchmark")
    table.add_column("Sequence Length", justify="right")
    table.add_column("Attention Type", justify="center")
    table.add_column("Window Size", justify="center")
    table.add_column("Tokens/sec", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Memory Savings", justify="right")
    
    # Benchmark each sequence length
    for seq_length in args.seq_lengths:
        console.print(f"Benchmarking sequence length: {seq_length}")
        
        # Update max_positions in model args
        model_args.max_positions = seq_length
        
        # Create dataset
        examples = create_random_dataset(tokenizer, seq_length, args.num_samples)
        dataset = MPNetDataset(tokenizer=tokenizer, dataset=examples, block_size=seq_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            collate_fn=collator,
        )
        
        # Benchmark standard model
        try:
            standard_results = benchmark_model(standard_model, dataloader, device)
            console.print(f"Standard model: {standard_results['tokens_per_sec']:.2f} tokens/sec, "
                         f"{standard_results['memory_mb']:.2f} MB")
            
            # Add to table
            table.add_row(
                str(seq_length),
                "Standard",
                "N/A",
                f"{standard_results['tokens_per_sec']:.2f}",
                f"{standard_results['memory_mb']:.2f}",
                "1.00x",
                "100%",
            )
            
            # Create FlexAttention model with different window sizes
            for window_size in args.window_sizes:
                if window_size >= seq_length:
                    continue  # Skip window sizes larger than sequence length
                
                # Update sliding window size
                model_args.sliding_window_size = window_size
                
                # Create flex model
                flex_model = MPNetFlexForPretraining(model_args, tokenizer)
                flex_model.to(device)
                
                # Benchmark flex model
                flex_results = benchmark_model(flex_model, dataloader, device, use_flex_attention=True)
                
                # Calculate speedup and memory savings
                speedup = flex_results['tokens_per_sec'] / standard_results['tokens_per_sec']
                memory_savings = flex_results['memory_mb'] / standard_results['memory_mb'] * 100
                
                console.print(f"FlexAttention (window={window_size}): "
                            f"{flex_results['tokens_per_sec']:.2f} tokens/sec, "
                            f"{flex_results['memory_mb']:.2f} MB, "
                            f"Speedup: {speedup:.2f}x, "
                            f"Memory: {memory_savings:.2f}%")
                
                # Add to table
                table.add_row(
                    str(seq_length),
                    "FlexAttention",
                    str(window_size),
                    f"{flex_results['tokens_per_sec']:.2f}",
                    f"{flex_results['memory_mb']:.2f}",
                    f"{speedup:.2f}x",
                    f"{memory_savings:.2f}%",
                )
                
                # Clean up
                del flex_model
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            # Probably out of memory
            console.print(f"[red]Error for sequence length {seq_length}: {e}[/red]")
            table.add_row(
                str(seq_length),
                "Standard",
                "N/A",
                "OOM",
                "OOM",
                "-",
                "-",
            )
    
    # Print final results
    console.print(table)


if __name__ == "__main__":
    main()