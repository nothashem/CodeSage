#!/usr/bin/env python3
"""
Training script for the Transformer model.
"""
import os
import json
import logging
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import wandb
import numpy as np

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.model.transformer import TransformerModel
from src.model.config import ModelConfig
from src.tokenizer.subword import BPETokenizer, WordPieceTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    """Dataset for loading and tokenizing code data."""
    
    def __init__(self, data_path: str, tokenizer: WordPieceTokenizer, max_length: int = 512, use_small_dataset: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.data_path = data_path
        
        # Use smaller dataset for testing
        if use_small_dataset:
            data_path_str = str(data_path)
            if 'train.txt' in data_path_str:
                data_path = data_path_str.replace('train.txt', 'train_small.txt')
            elif 'validation.txt' in data_path_str:
                data_path = data_path_str.replace('validation.txt', 'validation_small.txt')
            logger.info(f"Using small dataset: {data_path}")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        logger.info(f"Loading {len(lines)} lines from {data_path}")
        print("Starting tokenization loop...")
        # Tokenize and filter data
        for i, line in enumerate(tqdm(lines, desc="Tokenizing data")):
            if i == 0:
                print(f"Processing first line: '{line[:50]}...'")
            line = line.strip()
            if len(line) > 10:  # Skip very short lines
                try:
                    print(f"About to encode line {i}...")
                    tokens = self.tokenizer.encode(line)
                    print(f"Successfully encoded line {i} into {len(tokens)} tokens")
                    if len(tokens) > 1:  # Skip single token sequences
                        self.data.append(tokens)
                except Exception as e:
                    print(f"Error tokenizing line {i}: {e}")
                    print(f"Line content: '{line[:100]}...'")
                    continue
            if i % 100 == 0:
                print(f"Tokenized {i} lines")
        print("Finished tokenization loop.")
        
        logger.info(f"Loaded {len(self.data)} sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad with 0
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids, target_ids = zip(*batch)
    return torch.stack(input_ids), torch.stack(target_ids)

class Trainer:
    """Training class for the Transformer model."""
    
    def __init__(self, config: ModelConfig, tokenizer_path: str, data_dir: str, no_wandb: bool = False):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.no_wandb = no_wandb
        
        # Load tokenizer with fallback
        self.tokenizer = self._load_tokenizer_with_fallback(tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {len(self.tokenizer)}")
        
        # Update config with actual vocab size
        self.config.vocab_size = len(self.tokenizer)
        
        # Initialize model
        self.model = TransformerModel(config).to(self.device)
        logger.info(f"Model parameters: {self.model.get_num_params():,}")
        
        # Setup data
        self.data_dir = Path(data_dir)
        self.train_dataset = CodeDataset(
            self.data_dir / "train.txt",
            self.tokenizer,
            config.max_length,
            use_small_dataset=getattr(config, 'use_small_dataset', True)  # Use small dataset by default
        )
        self.val_dataset = CodeDataset(
            self.data_dir / "validation.txt",
            self.tokenizer,
            config.max_length,
            use_small_dataset=getattr(config, 'use_small_dataset', True)  # Use small dataset by default
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Reduce from 4 to 0 to save memory
            pin_memory=False  # Disable pin_memory to save memory
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,  # Reduce from 4 to 0 to save memory
            pin_memory=False  # Disable pin_memory to save memory
        )
        
        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Cosine learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=100,  # Will be updated based on actual epochs
            steps_per_epoch=len(self.train_loader),
            pct_start=config.warmup_steps / (100 * len(self.train_loader))
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
        # Training state
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def _load_tokenizer_with_fallback(self, tokenizer_path: str):
        """Load tokenizer with fallback for old format files."""
        try:
            # Try to load with the standard method
            try:
                return BPETokenizer.load(tokenizer_path)
            except Exception:
                return WordPieceTokenizer.load(tokenizer_path)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            logger.info("Creating simple character-level fallback tokenizer...")
            
            # Load the old format and create a simple character-level tokenizer
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create a simple character-level tokenizer
            vocab = data.get('vocab', {})
            if not vocab:
                raise ValueError("No vocabulary found in tokenizer file")
            
            # Create a simple tokenizer class
            class SimpleTokenizer:
                def __init__(self, vocab, max_length):
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
                    if isinstance(tokens[0], int):
                        tokens = [self.id_to_token.get(idx, ' ') for idx in tokens]
                    return ''.join(tokens)
                
                def __len__(self):
                    return len(self.token_to_id)
            
            tokenizer = SimpleTokenizer(vocab, self.config.max_length)
            logger.info(f"Created simple fallback tokenizer with {len(tokenizer)} tokens")
            return tokenizer
    
    def _wandb_log(self, *args, **kwargs):
        if not self.no_wandb:
            wandb.log(*args, **kwargs)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(input_ids)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = self.criterion(logits, target_ids)
            
            # Backward pass
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if self.global_step % 100 == 0:
                self._wandb_log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'global_step': self.global_step
                })
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc="Validating"):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits, _ = self.model(input_ids)
                logits = logits.view(-1, logits.size(-1))
                target_ids = target_ids.view(-1)
                
                loss = self.criterion(logits, target_ids)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best_checkpoint.pt")
            logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def generate_sample(self, prompt: str, max_length: int = 100):
        """Generate a sample from the model."""
        self.model.eval()
        
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p
            )
        
        # Decode
        generated_tokens = generated[0].cpu().numpy()
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def train(self, num_epochs: int):
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Log metrics
            self._wandb_log({
                'epoch': epoch,
                'train_loss_epoch': train_loss,
                'val_loss': val_loss
            })
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Generate sample
            if epoch % 5 == 0:
                sample = self.generate_sample("def hello_world():")
                logger.info(f"Sample generation:\n{sample}")
                self._wandb_log({'sample_generation': sample})

def main():
    parser = argparse.ArgumentParser(description="Train Transformer model")
    parser.add_argument("--tokenizer", default="trained_tokenizer.json", help="Path to trained tokenizer")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--use-small-dataset", action="store_true", default=True, help="Use small dataset for testing")
    parser.add_argument("--use-full-dataset", action="store_true", help="Use full dataset (overrides small dataset)")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="transformer-code-generation",
            config=vars(args)
        )
    
    # Create config
    config = ModelConfig(
        vocab_size=32000,  # Will be updated with actual vocab size
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_length=args.max_length,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Add dataset choice to config
    config.use_small_dataset = args.use_small_dataset and not args.use_full_dataset
    
    # Initialize trainer
    trainer = Trainer(config, args.tokenizer, args.data_dir, args.no_wandb)
    
    # Train
    trainer.train(args.epochs)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 