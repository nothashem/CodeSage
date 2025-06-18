import pytest
from src.bpe_tokenizer import BPETokenizer

def test_bpe_tokenizer_basic():
    # Create a small training corpus
    texts = [
        "hello world",
        "hello there",
        "world of tokens",
        "tokenization is fun"
    ]
    
    # Initialize and train the tokenizer
    tokenizer = BPETokenizer(vocab_size=50, min_freq=1)
    tokenizer.train(texts)
    
    # Test encoding
    test_text = "hello world"
    token_ids = tokenizer.encode(test_text)
    assert len(token_ids) > 0, "Encoding should produce token IDs"
    
    # Test decoding
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == test_text, "Decoding should recover original text"
    
    # Test save and load
    tokenizer.save("test_tokenizer.json")
    loaded_tokenizer = BPETokenizer.load("test_tokenizer.json")
    
    # Verify loaded tokenizer produces same results
    assert loaded_tokenizer.encode(test_text) == token_ids, "Loaded tokenizer should produce same encoding"
    
    # Test with unknown tokens
    unknown_text = "xyzabc"
    unknown_ids = tokenizer.encode(unknown_text)
    assert len(unknown_ids) > 0, "Should handle unknown tokens"
    
    # Clean up
    import os
    os.remove("test_tokenizer.json")

def test_bpe_tokenizer_merges():
    # Test with a corpus that should produce specific merges
    texts = [
        "low",
        "lower",
        "lowest",
        "new",
        "newer",
        "newest"
    ]
    
    tokenizer = BPETokenizer(vocab_size=30, min_freq=1)
    tokenizer.train(texts)
    
    # The tokenizer should learn to merge "er" and "est"
    assert any("er" in token for token in tokenizer.vocab.keys()), "Should learn 'er' merge"
    assert any("est" in token for token in tokenizer.vocab.keys()), "Should learn 'est' merge"
    
    # Test encoding preserves word boundaries
    test_text = "lower newest"
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == test_text, "Should preserve word boundaries"

def test_bpe_tokenizer_edge_cases():
    tokenizer = BPETokenizer(vocab_size=10, min_freq=1)
    
    # Test empty input
    assert tokenizer.encode("") == [], "Should handle empty string"
    assert tokenizer.decode([]) == "", "Should handle empty token list"
    
    # Test single character
    assert len(tokenizer.encode("a")) == 1, "Should handle single character"
    
    # Test with special characters
    special_text = "hello! world?"
    tokenizer.train([special_text])
    assert tokenizer.decode(tokenizer.encode(special_text)) == special_text, "Should handle special characters" 