import argparse
import json
from pathlib import Path
from typing import List, Dict, Union
from bpe_tokenizer import BPETokenizer
import sys
from datetime import datetime
import os

class TokenizerCLI:
    def __init__(self):
        self.tokenizer = None
        self.compression_stats = {
            "total_chars": 0,
            "total_tokens": 0,
            "compressed_size": 0,
            "original_size": 0
        }
    
    def train(self, data_path: str, vocab_size: int, min_freq: int, output_path: str):
        """Train a new tokenizer on the given data."""
        print(f"Loading training data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Training tokenizer with {len(texts)} examples...")
        self.tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq)
        self.tokenizer.train(texts)
        
        print(f"Saving tokenizer to {output_path}...")
        self.tokenizer.save(output_path)
        
        # Print training statistics
        print("\nTraining Statistics:")
        print(f"Vocabulary size: {len(self.tokenizer.vocab)}")
        print(f"Number of merges: {len(self.tokenizer.merges)}")
        
        # Show some example encodings
        print("\nExample encodings:")
        for text in texts[:3]:
            self._show_tokenization(text)
    
    def load(self, tokenizer_path: str):
        """Load an existing tokenizer."""
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer with {len(self.tokenizer.vocab)} tokens")
    
    def compress_file(self, input_path: str, output_path: str, tokenizer_path: str = None):
        """Compress a text file using the tokenizer."""
        if not self.tokenizer:
            if tokenizer_path and os.path.exists(tokenizer_path):
                print(f"Loading tokenizer from {tokenizer_path}...")
                self.load(tokenizer_path)
            else:
                print("Error: No tokenizer loaded and no valid tokenizer path provided.")
                print("Please either:")
                print("1. Train a new tokenizer using the 'train' command")
                print("2. Load an existing tokenizer using the 'load' command")
                print("3. Provide a valid tokenizer path when compressing")
                return
        
        print(f"Compressing {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Encode the text
        token_ids = self.tokenizer.encode(text)
        
        # Convert token IDs to bytes for more efficient storage
        # We'll use a variable-length encoding where:
        # - Numbers < 128 are stored as single bytes
        # - Larger numbers are stored as multiple bytes with a continuation bit
        def encode_number(n):
            if n < 128:
                return bytes([n])
            result = bytearray()
            while n > 0:
                byte = n & 0x7F  # Get 7 bits
                n >>= 7  # Shift right by 7 bits
                if n > 0:
                    byte |= 0x80  # Set continuation bit
                result.append(byte)
            return bytes(result)
        
        # Encode all token IDs
        encoded_bytes = bytearray()
        for token_id in token_ids:
            encoded_bytes.extend(encode_number(token_id))
        
        # Save compressed data
        compressed_data = {
            "data": encoded_bytes.hex(),  # Store as hex string
            "original_size": len(text.encode('utf-8')),
            "compressed_size": len(encoded_bytes),
            "num_tokens": len(token_ids),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(compressed_data, f)
        
        # Update statistics
        self.compression_stats["total_chars"] += len(text)
        self.compression_stats["total_tokens"] += len(token_ids)
        self.compression_stats["original_size"] += compressed_data["original_size"]
        self.compression_stats["compressed_size"] += compressed_data["compressed_size"]
        
        # Print compression statistics
        self._print_compression_stats(compressed_data)
    
    def decompress_file(self, input_path: str, output_path: str):
        """Decompress a compressed file back to text."""
        if not self.tokenizer:
            print("Error: No tokenizer loaded. Please load a tokenizer first.")
            return
        
        print(f"Decompressing {input_path}...")
        with open(input_path, 'r') as f:
            compressed_data = json.load(f)
        
        # Decode the compressed bytes
        def decode_number(byte_iter):
            result = 0
            shift = 0
            while True:
                byte = next(byte_iter)
                result |= (byte & 0x7F) << shift
                if not (byte & 0x80):
                    break
                shift += 7
            return result
        
        # Convert hex string back to bytes
        encoded_bytes = bytes.fromhex(compressed_data["data"])
        byte_iter = iter(encoded_bytes)
        
        # Decode all token IDs
        token_ids = []
        for _ in range(compressed_data["num_tokens"]):
            token_ids.append(decode_number(byte_iter))
        
        # Decode the token IDs to text
        text = self.tokenizer.decode(token_ids)
        
        # Save decompressed text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Decompressed text saved to {output_path}")
        
        # Verify the decompression
        original_size = compressed_data["original_size"]
        decompressed_size = len(text.encode('utf-8'))
        if original_size != decompressed_size:
            print(f"Warning: Decompressed size ({decompressed_size}) differs from original size ({original_size})")
        else:
            print("Decompression verified: sizes match")
    
    def analyze_text(self, text: str):
        """Analyze tokenization of a given text."""
        if not self.tokenizer:
            print("Error: No tokenizer loaded. Please load or train a tokenizer first.")
            return
        
        self._show_tokenization(text)
    
    def _show_tokenization(self, text: str):
        """Show detailed tokenization of a text."""
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.inverse_vocab[tid] for tid in token_ids]
        
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Number of characters: {len(text)}")
        print(f"Compression ratio: {len(text) / len(tokens):.2f} chars per token")
    
    def _print_compression_stats(self, stats: Dict):
        """Print compression statistics."""
        print("\nCompression Statistics:")
        print(f"Original size: {stats['original_size']} bytes")
        print(f"Compressed size: {stats['compressed_size']} bytes")
        compression_ratio = stats['original_size'] / stats['compressed_size']
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print(f"Space saved: {(1 - 1/compression_ratio) * 100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="BPE Tokenizer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new tokenizer")
    train_parser.add_argument("--data", required=True, help="Path to training data file")
    train_parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size")
    train_parser.add_argument("--min-freq", type=int, default=2, help="Minimum token frequency")
    train_parser.add_argument("--output", required=True, help="Output path for the tokenizer")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load an existing tokenizer")
    load_parser.add_argument("--tokenizer", required=True, help="Path to the tokenizer file")
    
    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress a text file")
    compress_parser.add_argument("--input", required=True, help="Input text file to compress")
    compress_parser.add_argument("--output", required=True, help="Output compressed file")
    compress_parser.add_argument("--tokenizer", help="Path to the tokenizer file (optional if already loaded)")
    
    # Decompress command
    decompress_parser = subparsers.add_parser("decompress", help="Decompress a file")
    decompress_parser.add_argument("--input", required=True, help="Input compressed file")
    decompress_parser.add_argument("--output", required=True, help="Output text file")
    decompress_parser.add_argument("--tokenizer", help="Path to the tokenizer file (optional if already loaded)")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text tokenization")
    analyze_parser.add_argument("--text", required=True, help="Text to analyze")
    analyze_parser.add_argument("--tokenizer", help="Path to the tokenizer file (optional if already loaded)")
    
    args = parser.parse_args()
    cli = TokenizerCLI()
    
    if args.command == "train":
        cli.train(args.data, args.vocab_size, args.min_freq, args.output)
    elif args.command == "load":
        cli.load(args.tokenizer)
    elif args.command == "compress":
        cli.compress_file(args.input, args.output, args.tokenizer)
    elif args.command == "decompress":
        if not cli.tokenizer and args.tokenizer:
            cli.load(args.tokenizer)
        cli.decompress_file(args.input, args.output)
    elif args.command == "analyze":
        if not cli.tokenizer and args.tokenizer:
            cli.load(args.tokenizer)
        cli.analyze_text(args.text)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 