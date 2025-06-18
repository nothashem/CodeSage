"""
Base tokenizer classes and common utilities for the unified tokenizer hierarchy.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Any
import json
from pathlib import Path


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizer implementations.
    
    This class defines the common interface that all tokenizers must implement,
    along with shared functionality for vocabulary management and file I/O.
    """
    
    # Standardized special tokens
    SPECIAL_TOKENS = {
        "pad": "[PAD]",
        "unk": "[UNK]", 
        "bos": "[BOS]",
        "eos": "[EOS]",
        "mask": "[MASK]",
        "sep": "[SEP]",
        "cls": "[CLS]"
    }
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        special_tokens: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the base tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            special_tokens: Additional special tokens to add
            **kwargs: Additional tokenizer-specific parameters
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Initialize special tokens
        self.special_tokens = list(self.SPECIAL_TOKENS.values())
        if special_tokens:
            self.special_tokens.extend(special_tokens)
            
        # Initialize vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Initialize special tokens in vocabulary
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self) -> None:
        """Initialize the special tokens in the vocabulary."""
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    @abstractmethod
    def encode(self, text: str) -> List[str]:
        """
        Convert text to tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        pass
    
    @abstractmethod
    def encode_ids(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, tokens: Union[List[str], List[int]]) -> str:
        """
        Convert tokens or token IDs back to text.
        
        Args:
            tokens: List of tokens or token IDs
            
        Returns:
            Decoded text
        """
        pass
    
    @abstractmethod
    def train(self, texts: List[str], **kwargs) -> None:
        """
        Train the tokenizer on text corpus.
        
        Args:
            texts: List of training texts
            **kwargs: Training-specific parameters
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            path: Path to save the tokenizer
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer state
        state = {
            "tokenizer_type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "special_tokens": self.special_tokens,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "tokenizer_specific": self._get_tokenizer_specific_state()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BaseTokenizer':
        """
        Load tokenizer from file.
        
        Args:
            path: Path to the saved tokenizer
            
        Returns:
            Loaded tokenizer instance
        """
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=state["vocab_size"],
            max_length=state["max_length"],
            special_tokens=state["special_tokens"]
        )
        
        # Restore vocabulary
        tokenizer.token_to_id = state["token_to_id"]
        tokenizer.id_to_token = state["id_to_token"]
        
        # Restore tokenizer-specific state
        tokenizer._restore_tokenizer_specific_state(state.get("tokenizer_specific", {}))
        
        return tokenizer
    
    def _get_tokenizer_specific_state(self) -> Dict[str, Any]:
        """
        Get tokenizer-specific state for saving.
        Override in subclasses to save additional state.
        
        Returns:
            Dictionary of tokenizer-specific state
        """
        return {}
    
    def _restore_tokenizer_specific_state(self, state: Dict[str, Any]) -> None:
        """
        Restore tokenizer-specific state from saved state.
        Override in subclasses to restore additional state.
        
        Args:
            state: Dictionary of tokenizer-specific state
        """
        pass
    
    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.token_to_id)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size."""
        return len(self.token_to_id)
    
    def add_special_tokens(self, tokens: List[str]) -> None:
        """
        Add additional special tokens to the vocabulary.
        
        Args:
            tokens: List of special tokens to add
        """
        for token in tokens:
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
                self.special_tokens.append(token)
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of tokenized texts
        """
        return [self.encode(text) for text in texts]
    
    def tokenize_batch_ids(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize a batch of texts to token IDs.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            List of token ID sequences
        """
        return [self.encode_ids(text) for text in texts] 