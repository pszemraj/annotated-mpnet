"""
Test script for FlexAttention implementation for MPNet.
"""
import torch
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from annotated_mpnet.transformer_modules.flex_attention import FlexTwoStreamAttention
from annotated_mpnet.modeling.mpnet_for_pretraining import MPNetForPretraining

def test_flex_two_stream_attention():
    """Test basic functionality of FlexTwoStreamAttention."""
    print("Testing FlexTwoStreamAttention...")
    # Initialize model
    model = FlexTwoStreamAttention(
        embed_dim=768,
        num_heads=12,
        dropout=0.1
    )
    
    # Create dummy inputs
    seq_len = 16
    batch_size = 4
    embed_dim = 768
    
    # Create dummy content and query tensors
    content = torch.randn(seq_len, batch_size, embed_dim)
    query = torch.randn(seq_len//2, batch_size, embed_dim)
    key = content
    value = content
    
    # Forward pass
    print("Running forward pass...")
    outputs, _ = model((content, query), key, value)
    c_out, q_out = outputs
    
    print(f"Content output shape: {c_out.shape}")
    print(f"Query output shape: {q_out.shape}")
    
    # Verify output shapes
    assert c_out.shape == content.shape, f"Expected {content.shape}, got {c_out.shape}"
    assert q_out.shape == query.shape, f"Expected {query.shape}, got {q_out.shape}"
    
    print("FlexTwoStreamAttention test passed!")
    return True

def test_mpnet_with_flex_attention():
    """Test the MPNet model with FlexAttention enabled."""
    print("Testing MPNet with FlexAttention...")
    
    class Args:
        def __init__(self):
            self.encoder_layers = 2
            self.encoder_embed_dim = 768
            self.encoder_ffn_dim = 3072
            self.encoder_attention_heads = 12
            self.dropout = 0.1
            self.attention_dropout = 0.1
            self.activation_dropout = 0.1
            self.max_positions = 512
            self.activation_fn = "gelu"
            self.normalize_before = False
            self.use_flex_attention = True
            self.sliding_window_size = None
    
    class DummyTokenizer:
        def __init__(self):
            self.vocab = {"[PAD]": 0}
            self.pad_token = "[PAD]"
            self.vocab_size = 30000
    
    args = Args()
    tokenizer = DummyTokenizer()
    
    # Initialize model
    print("Initializing MPNetForPretraining...")
    model = MPNetForPretraining(args, tokenizer)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 16
    pred_size = 4
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    print("Running forward pass...")
    try:
        outputs = model(
            input_ids=input_ids,
            positions=positions,
            pred_size=pred_size
        )
        print(f"Output shape: {outputs.shape}")
        print("MPNet with FlexAttention test passed!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_flex_two_stream_attention():
        test_mpnet_with_flex_attention()