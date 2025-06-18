"""
Test cases for the tokenizer implementation.
"""
import pytest
from src.tokenizer import Tokenizer
import tempfile
import json
import os

@pytest.fixture
def tokenizer():
    """Create a tokenizer instance for testing."""
    return Tokenizer()

@pytest.fixture
def vocab_file():
    """Create a temporary vocabulary file for testing."""
    vocab = {
        "hello": 4,
        "world": 5,
        "!": 6,
        "test": 7,
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(vocab, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

def test_initialization(tokenizer):
    """Test tokenizer initialization."""
    assert len(tokenizer.special_tokens) == 4
    assert tokenizer.PAD_TOKEN in tokenizer.token_to_id
    assert tokenizer.UNK_TOKEN in tokenizer.token_to_id
    assert tokenizer.BOS_TOKEN in tokenizer.token_to_id
    assert tokenizer.EOS_TOKEN in tokenizer.token_to_id

def test_vocab_loading(tokenizer, vocab_file):
    """Test vocabulary loading from file."""
    tokenizer.load_vocab(vocab_file)
    
    # Check if vocabulary was loaded correctly
    assert "hello" in tokenizer.token_to_id
    assert "world" in tokenizer.token_to_id
    assert tokenizer.token_to_id["hello"] == 4
    assert tokenizer.token_to_id["world"] == 5
    
    # Check if special tokens are preserved
    assert tokenizer.PAD_TOKEN in tokenizer.token_to_id
    assert tokenizer.UNK_TOKEN in tokenizer.token_to_id

def test_encoding(tokenizer, vocab_file):
    """Test text encoding."""
    tokenizer.load_vocab(vocab_file)
    
    # Test encoding with known tokens
    tokens = tokenizer.encode("hello world!")
    assert tokens == ["hello", "world", "!"]
    
    # Test encoding with unknown tokens
    tokens = tokenizer.encode("hello unknown world!")
    assert tokens == ["hello", "[UNK]", "world", "!"]

def test_encoding_ids(tokenizer, vocab_file):
    """Test text encoding to IDs."""
    tokenizer.load_vocab(vocab_file)
    
    # Test encoding with known tokens
    ids = tokenizer.encode_ids("hello world!")
    assert ids == [4, 5, 6]
    
    # Test encoding with unknown tokens
    ids = tokenizer.encode_ids("hello unknown world!")
    unk_id = tokenizer.token_to_id[tokenizer.UNK_TOKEN]
    assert ids == [4, unk_id, 5, 6]

def test_decoding(tokenizer, vocab_file):
    """Test token decoding."""
    tokenizer.load_vocab(vocab_file)
    
    # Test decoding from tokens
    text = "hello world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    assert decoded == text
    
    # Test decoding from IDs
    ids = tokenizer.encode_ids(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text
    
    # Test decoding with special tokens
    tokens = [tokenizer.BOS_TOKEN, "hello", "world", tokenizer.EOS_TOKEN]
    decoded = tokenizer.decode(tokens)
    assert decoded == "hello world"

def test_save_vocab(tokenizer, vocab_file):
    """Test vocabulary saving."""
    # Load initial vocabulary
    tokenizer.load_vocab(vocab_file)
    
    # Save to new file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        save_path = f.name
    
    try:
        tokenizer.save_vocab(save_path)
        
        # Load saved vocabulary
        new_tokenizer = Tokenizer()
        new_tokenizer.load_vocab(save_path)
        
        # Compare vocabularies
        assert new_tokenizer.token_to_id == tokenizer.token_to_id
    finally:
        os.unlink(save_path) 