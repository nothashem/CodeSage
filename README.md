# Python LLM Tokenizer

A simple and efficient tokenizer implementation for Large Language Models (LLMs) in Python. This project provides a flexible tokenization system that supports both word-level and subword tokenization, with a focus on performance and ease of use.

## Features

- Word-level and subword tokenization
- Efficient vocabulary management
- Special token handling ([PAD], [UNK], [BOS], [EOS])
- Batch processing support
- Type hints and comprehensive documentation
- Extensive test coverage

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd python-llm-tokenizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.tokenizer import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer(vocab_path="path/to/vocab.json")

# Tokenize text
tokens = tokenizer.encode("Hello, world!")
ids = tokenizer.encode_ids("Hello, world!")

# Decode tokens
text = tokenizer.decode(tokens)
```

## Project Structure

```
.
├── src/
│   ├── tokenizer.py    # Core tokenizer implementation
│   └── vocab.py        # Vocabulary management
├── tests/
│   └── test_tokenizer.py
├── requirements.txt
└── README.md
```

## Development

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest tests/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 