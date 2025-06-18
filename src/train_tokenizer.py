from bpe_tokenizer import BPETokenizer
from pathlib import Path
import json
from typing import List
import tqdm

def load_text_data(file_path: str) -> List[str]:
    """Load text data from a file, one line per example."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def train_tokenizer(
    data_path: str,
    vocab_size: int = 1000,
    min_freq: int = 2,
    output_path: str = "tokenizer.json"
):
    """
    Train a BPE tokenizer on the given data.
    
    Args:
        data_path: Path to the training data file (one example per line)
        vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency for a token to be considered
        output_path: Where to save the trained tokenizer
    """
    print(f"Loading data from {data_path}...")
    texts = load_text_data(data_path)
    print(f"Loaded {len(texts)} training examples")
    
    # Initialize and train the tokenizer
    print(f"Training tokenizer with vocab_size={vocab_size}, min_freq={min_freq}...")
    tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.train(texts)
    
    # Save the trained tokenizer
    print(f"Saving tokenizer to {output_path}...")
    tokenizer.save(output_path)
    
    # Print some statistics
    print("\nTokenizer Statistics:")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Number of merges learned: {len(tokenizer.merges)}")
    
    # Print some example encodings
    print("\nExample encodings:")
    for text in texts[:3]:  # Show first 3 examples
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)
        print(f"\nOriginal: {text}")
        print(f"Token IDs: {token_ids}")
        print(f"Decoded: {decoded}")
        print(f"Number of tokens: {len(token_ids)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--data", required=True, help="Path to training data file")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Maximum vocabulary size")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--output", default="tokenizer.json", help="Output path for the trained tokenizer")
    
    args = parser.parse_args()
    train_tokenizer(
        data_path=args.data,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        output_path=args.output
    ) 