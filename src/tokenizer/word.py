"""
Word-level tokenizer implementation.
"""
import regex as re
from typing import List, Dict, Optional, Union
from .base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    Simple word-level tokenizer that splits text on whitespace and punctuation.
    
    This tokenizer treats each word as a separate token, making it suitable
    for languages with clear word boundaries or as a baseline for comparison.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        special_tokens: Optional[List[str]] = None,
        lowercase: bool = True,
        remove_punctuation: bool = False,
    ):
        """
        Initialize the word tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            special_tokens: Additional special tokens to add
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation tokens
        """
        super().__init__(vocab_size, max_length, special_tokens)
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
    
    def encode(self, text: str) -> List[str]:
        """
        Convert text to word tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        if self.lowercase:
            text = text.lower()
        
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            words = [word for word in words if word.isalnum()]
        
        # Replace unknown words with UNK token
        return [word if word in self.token_to_id else self.SPECIAL_TOKENS["unk"] for word in words]
    
    def encode_ids(self, text: str) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        tokens = self.encode(text)
        return [self.token_to_id.get(token, self.token_to_id[self.SPECIAL_TOKENS["unk"]]) for token in tokens]
    
    def decode(self, tokens: Union[List[str], List[int]]) -> str:
        """
        Convert tokens or token IDs back to text.
        
        Args:
            tokens: List of tokens or token IDs
            
        Returns:
            Decoded text
        """
        if not tokens:
            return ""
            
        # Convert token IDs to tokens if necessary
        if isinstance(tokens[0], int):
            tokens = [self.id_to_token.get(idx, self.SPECIAL_TOKENS["unk"]) for idx in tokens]
            
        # Remove special tokens
        tokens = [t for t in tokens if t not in self.special_tokens]
        
        # Join tokens with proper spacing
        result = []
        for i, token in enumerate(tokens):
            # Don't add space before punctuation
            if i > 0 and not (token in '.,!?;:') and not (tokens[i-1] in '.,!?;:'):
                result.append(' ')
            result.append(token)
        
        return ''.join(result)
    
    def train(self, texts: List[str], **kwargs) -> None:
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of training texts
            **kwargs: Additional training parameters (ignored for word tokenizer)
        """
        # Count word frequencies
        word_counts = {}
        for text in texts:
            tokens = self.encode(text)
            for token in tokens:
                if token not in self.special_tokens:
                    word_counts[token] = word_counts.get(token, 0) + 1
        
        # Sort by frequency and take top vocab_size tokens
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add words to vocabulary (special tokens are already added)
        for word, _ in sorted_words[:self.vocab_size - len(self.special_tokens)]:
            if word not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[word] = idx
                self.id_to_token[idx] = word
    
    def _get_tokenizer_specific_state(self) -> Dict[str, any]:
        """Get word tokenizer specific state."""
        return {
            "lowercase": self.lowercase,
            "remove_punctuation": self.remove_punctuation
        }
    
    def _restore_tokenizer_specific_state(self, state: Dict[str, any]) -> None:
        """Restore word tokenizer specific state."""
        self.lowercase = state.get("lowercase", True)
        self.remove_punctuation = state.get("remove_punctuation", False) 