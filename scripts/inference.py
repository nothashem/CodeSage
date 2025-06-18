#!/usr/bin/env python3
"""
Inference script for the Transformer model.
"""
import argparse
import torch
import json
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model.transformer import TransformerModel
from src.model.config import ModelConfig
from src.tokenizer.subword import BPETokenizer, WordPieceTokenizer

def load_tokenizer_with_fallback(tokenizer_path: str):
    """Load tokenizer with fallback for old format files."""
    try:
        # Try to load with the standard method
        try:
            return BPETokenizer.load(tokenizer_path)
        except Exception:
            return WordPieceTokenizer.load(tokenizer_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        print("Creating simple character-level fallback tokenizer...")
        
        # Load the old format and create a simple character-level tokenizer
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a simple character-level tokenizer
        vocab = data.get('vocab', {})
        if not vocab:
            raise ValueError("No vocabulary found in tokenizer file")
        
        # Create a simple tokenizer class
        class SimpleTokenizer:
            def __init__(self, vocab, max_length=512):
                self.token_to_id = vocab
                self.id_to_token = {v: k for k, v in vocab.items()}
                self.max_length = max_length
            
            def encode(self, text):
                # Simple character-level tokenization, return token IDs
                tokens = []
                for char in text:
                    tokens.append(self.token_to_id.get(char, self.token_to_id.get(' ', 0)))
                return tokens
            
            def encode_ids(self, text):
                return self.encode(text)
            
            def decode(self, tokens):
                # Convert numpy.int64 to int if needed
                tokens = [int(t) if not isinstance(t, str) else t for t in tokens]
                if isinstance(tokens[0], int):
                    tokens = [self.id_to_token.get(idx, ' ') for idx in tokens]
                return ''.join(tokens)
            
            def __len__(self):
                return len(self.token_to_id)
        
        tokenizer = SimpleTokenizer(vocab)
        print(f"Created simple fallback tokenizer with {len(tokenizer)} tokens")
        return tokenizer

def load_model(checkpoint_path: str, tokenizer_path: str):
    """Load a trained model from checkpoint."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    # Load tokenizer with fallback
    tokenizer = load_tokenizer_with_fallback(tokenizer_path)
    
    # Initialize model
    model = TransformerModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, tokenizer, config

def create_demo_model(tokenizer_path: str):
    """Create a demo model for testing without training."""
    # Load tokenizer with fallback
    tokenizer = load_tokenizer_with_fallback(tokenizer_path)
    
    # Create a simple config
    config = ModelConfig(
        vocab_size=len(tokenizer),
        d_model=256,  # Smaller for demo
        n_heads=4,
        n_layers=2,
        d_ff=1024,
        max_length=128,
        dropout=0.1,
        batch_size=1,
        learning_rate=1e-4
    )
    
    # Initialize model
    model = TransformerModel(config)
    
    return model, tokenizer, config

def generate_text(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 0.7):
    """Generate text from a prompt."""
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
    
    # Decode
    generated_tokens = generated[0].numpy()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def interactive_mode(model, tokenizer, max_length: int = 100, temperature: float = 0.7):
    """Run interactive mode for testing queries."""
    print("\n=== Interactive Mode ===")
    print("Type your prompts and press Enter. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print(f"\nGenerating for: '{prompt}'")
            print("-" * 30)
            
            generated = generate_text(model, tokenizer, prompt, max_length, temperature)
            print(f"Generated: {generated}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="trained_tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--prompt", help="Input prompt")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--demo", action="store_true", help="Run with demo model (no training required)")
    
    args = parser.parse_args()
    
    # Load model
    if args.demo:
        print("Loading demo model...")
        model, tokenizer, config = create_demo_model(args.tokenizer)
        print("Note: This is an untrained demo model. Results will be random.")
    elif args.checkpoint and Path(args.checkpoint).exists():
        print("Loading trained model...")
        model, tokenizer, config = load_model(args.checkpoint, args.tokenizer)
    else:
        print("No checkpoint found. Using demo model...")
        model, tokenizer, config = create_demo_model(args.tokenizer)
        print("Note: This is an untrained demo model. Results will be random.")
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print("-" * 50)
    
    # Run interactive mode
    if args.interactive:
        interactive_mode(model, tokenizer, args.max_length, args.temperature)
        return
    
    # Single prompt mode
    if args.prompt:
        print(f"Input prompt: {args.prompt}")
        print("-" * 50)
        
        # Generate text
        generated = generate_text(model, tokenizer, args.prompt, args.max_length, args.temperature)
        
        print("Generated text:")
        print(generated)
    else:
        # Default demo
        demo_prompts = [
            "def hello_world():",
            "class Calculator:",
            "def calculate_sum(a, b):",
            "import numpy as np",
            "def fibonacci(n):"
        ]
        
        print("Running demo with sample prompts:")
        for prompt in demo_prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 30)
            generated = generate_text(model, tokenizer, prompt, args.max_length, args.temperature)
            print(f"Generated: {generated}")

if __name__ == "__main__":
    main() 