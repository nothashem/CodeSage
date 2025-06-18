"""
Subword tokenizer implementations including BPE and WordPiece.
"""
from abc import abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Union, Tuple, Set
import regex as re
from .base import BaseTokenizer


class SubwordTokenizer(BaseTokenizer):
    """
    Abstract base class for subword tokenizers.
    
    Subword tokenizers break words into smaller units (subwords) to handle
    out-of-vocabulary words and reduce vocabulary size while maintaining
    coverage of the language.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2,
        **kwargs
    ):
        """
        Initialize the subword tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            special_tokens: Additional special tokens to add
            min_frequency: Minimum frequency for a token to be considered
            **kwargs: Additional tokenizer-specific parameters
        """
        super().__init__(vocab_size, max_length, special_tokens, **kwargs)
        self.min_frequency = min_frequency
        self.merges: Dict[Tuple[str, str], str] = {}
    
    @abstractmethod
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using the subword algorithm.
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of subword tokens
        """
        pass
    
    def encode(self, text: str) -> List[str]:
        """
        Convert text to subword tokens.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of subword tokens
        """
        # Split text into words and tokenize each word
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        tokens = []
        
        for word in words:
            if word in self.special_tokens:
                tokens.append(word)
            else:
                word_tokens = self._tokenize_word(word)
                tokens.extend(word_tokens)
        
        return tokens
    
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
    
    def _get_tokenizer_specific_state(self) -> Dict[str, any]:
        """Get subword tokenizer specific state."""
        return {
            "min_frequency": self.min_frequency,
            "merges": {f"{pair[0]}_{pair[1]}": merged for pair, merged in self.merges.items()}
        }
    
    def _restore_tokenizer_specific_state(self, state: Dict[str, any]) -> None:
        """Restore subword tokenizer specific state."""
        self.min_frequency = state.get("min_frequency", 2)
        merges_dict = state.get("merges", {})
        self.merges = {}
        for key, merged in merges_dict.items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                self.merges[(parts[0], parts[1])] = merged


class BPETokenizer(SubwordTokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer implementation.
    
    BPE is a subword tokenization algorithm that iteratively merges the most
    frequent adjacent pairs of tokens in the training corpus.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2,
    ):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            special_tokens: Additional special tokens to add
            min_frequency: Minimum frequency for a token to be considered
        """
        super().__init__(vocab_size, max_length, special_tokens, min_frequency)
        self._init_base_vocab()
    
    def _init_base_vocab(self) -> None:
        """Initialize the base vocabulary with common characters."""
        # Add basic ASCII characters
        for i in range(32, 127):  # Printable ASCII characters
            char = chr(i)
            if char not in self.token_to_id:
                self.token_to_id[char] = len(self.token_to_id)
                self.id_to_token[len(self.token_to_id) - 1] = char
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using BPE algorithm.
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of BPE tokens
        """
        # Start with character-level tokens
        tokens = list(word)
        
        # Apply merges greedily
        max_iterations = len(tokens)  # Safety check to prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Get all possible pairs
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            if not pairs:
                break
            
            # Find the best pair to merge
            best_pair = None
            best_merge = None
            for pair in pairs:
                if pair in self.merges:
                    merged = self.merges[pair]
                    if best_pair is None or len(merged) > len(best_merge):
                        best_pair = pair
                        best_merge = merged
            
            if best_pair is None:
                break
            
            # Merge the best pair
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and 
                    tokens[i] == best_pair[0] and 
                    tokens[i + 1] == best_pair[1]):
                    new_tokens.append(best_merge)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            # Safety check: if no tokens were merged, break to prevent infinite loop
            if len(new_tokens) >= len(tokens):
                break
                
            tokens = new_tokens
        
        return tokens
    
    def train(self, texts: List[str], **kwargs) -> None:
        """
        Train the BPE tokenizer on the given texts.
        
        Args:
            texts: List of training texts
            **kwargs: Additional training parameters
        """
        # Preprocess texts into words
        words = []
        for text in texts:
            words.extend(re.findall(r'\b\w+\b', text))
        
        # Convert words to character-level tokens
        tokenized_words = [[char for char in word] for word in words]
        
        # Learn merges
        num_merges = self.vocab_size - len(self.token_to_id)
        for _ in range(num_merges):
            # Get statistics for current tokenization
            pairs = self._get_word_stats(tokenized_words)
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Only merge if the pair appears frequently enough
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Merge the pair in all words
            tokenized_words = [self._merge_pair_in_word(best_pair, word) 
                             for word in tokenized_words]
            
            # Update vocabulary and merges
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.token_to_id)
                self.id_to_token[len(self.token_to_id) - 1] = merged_token
            self.merges[best_pair] = merged_token
    
    def _get_word_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs within words."""
        pairs = defaultdict(int)
        for word in words:
            if len(word) <= 1:
                continue
            # Count pairs within the same word
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs
    
    def _merge_pair_in_word(self, pair: Tuple[str, str], word: List[str]) -> List[str]:
        """Merge a pair of tokens within a word."""
        if len(word) <= 1:
            return word
        
        new_word = []
        i = 0
        while i < len(word):
            if (i < len(word) - 1 and 
                word[i] == pair[0] and 
                word[i + 1] == pair[1]):
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word


class WordPieceTokenizer(SubwordTokenizer):
    """
    WordPiece tokenizer implementation based on BERT's tokenization algorithm.
    
    WordPiece uses a greedy longest-match-first approach with ## prefix for
    subword tokens, similar to BPE but with different scoring and merging strategy.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 512,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2,
        max_chars_per_token: int = 100,
    ):
        """
        Initialize the WordPiece tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            max_length: Maximum sequence length
            special_tokens: Additional special tokens to add
            min_frequency: Minimum frequency for a token to be considered
            max_chars_per_token: Maximum characters per token
        """
        super().__init__(vocab_size, max_length, special_tokens, min_frequency)
        self.max_chars_per_token = max_chars_per_token
        self.subword_prefix = "##"
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        Tokenize a single word using WordPiece algorithm.
        
        Args:
            word: Word to tokenize
            
        Returns:
            List of WordPiece tokens
        """
        if len(word) <= self.max_chars_per_token:
            # Try to find the word in vocabulary
            if word in self.token_to_id:
                return [word]
        
        # Use longest-match-first approach
        start = 0
        tokens = []
        
        while start < len(word):
            end = len(word)
            current_substr = None
            
            # Find the longest substring that exists in vocabulary
            while start < end:
                substr = word[start:end]
                if start == 0:
                    # First token doesn't get ## prefix
                    if substr in self.token_to_id:
                        current_substr = substr
                        break
                else:
                    # Subsequent tokens get ## prefix
                    prefixed_substr = self.subword_prefix + substr
                    if prefixed_substr in self.token_to_id:
                        current_substr = prefixed_substr
                        break
                end -= 1
            
            if current_substr is None:
                # If no match found, use the first character
                if start == 0:
                    current_substr = word[start:start + 1]
                else:
                    current_substr = self.subword_prefix + word[start:start + 1]
            
            tokens.append(current_substr)
            start = end
        
        return tokens
    
    def train(self, texts: List[str], **kwargs) -> None:
        """
        Train the WordPiece tokenizer on the given texts.
        
        Args:
            texts: List of training texts
            **kwargs: Additional training parameters
        """
        # This is a simplified WordPiece training implementation
        # In practice, WordPiece training is more complex and involves
        # likelihood maximization rather than frequency counting
        
        # For now, we'll use a BPE-like approach but with WordPiece scoring
        words = []
        for text in texts:
            words.extend(re.findall(r'\b\w+\b', text))
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Initialize vocabulary with characters
        for word in words:
            for char in word:
                if char not in self.token_to_id:
                    self.token_to_id[char] = len(self.token_to_id)
                    self.id_to_token[len(self.token_to_id) - 1] = char
        
        # Learn merges using WordPiece scoring
        num_merges = self.vocab_size - len(self.token_to_id)
        for _ in range(num_merges):
            # Get all possible pairs
            pairs = self._get_wordpiece_pairs(words, word_counts)
            if not pairs:
                break
            
            # Find best pair using WordPiece scoring
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            
            # Only merge if the pair appears frequently enough
            if pairs[best_pair] < self.min_frequency:
                break
            
            # Update vocabulary and merges
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = len(self.token_to_id)
                self.id_to_token[len(self.token_to_id) - 1] = merged_token
            self.merges[best_pair] = merged_token
    
    def _get_wordpiece_pairs(self, words: List[str], word_counts: Counter) -> Dict[Tuple[str, str], float]:
        """Get WordPiece pairs with scoring based on likelihood."""
        pairs = defaultdict(float)
        
        for word, count in word_counts.items():
            if count < self.min_frequency:
                continue
                
            # Get all possible pairs in this word
            word_pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            
            # Score pairs using WordPiece likelihood
            for pair in word_pairs:
                # Simplified scoring: frequency * pair probability
                pair_prob = word_pairs.count(pair) / len(word_pairs)
                pairs[pair] += count * pair_prob
        
        return pairs 