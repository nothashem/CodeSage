# CodeSage 🤖

A lightweight Seq2Seq Transformer model designed specifically for code understanding and generation tasks. CodeSage features a compact architecture with 19 million parameters, making it efficient and suitable for research, education, and deployment in resource-constrained environments.

## 🚀 Features

- **Bidirectional Code Processing**: Generate code from descriptions and descriptions from code
- **Lightweight Architecture**: Only 19M parameters for efficient inference
- **Interactive CLI**: User-friendly command-line interface
- **PyTorch Implementation**: Modern, maintainable codebase
- **Docker Support**: Ready for containerized deployment

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/codesage.git
   cd codesage
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### Interactive Interface

Run the interactive CLI to start generating code and descriptions:

```bash
python interactive_ui.py
```

This will launch a menu-driven interface where you can:
- Generate descriptions from code
- Generate code from descriptions
- Adjust generation parameters

### Programmatic Usage

```python
from src.model.layers import Seq2SeqTransformer
from interactive_ui import SimpleTokenizer

# Initialize model and tokenizer
model = Seq2SeqTransformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=128,
    n_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=512,
    dropout=0.1,
    max_length=64,
    padding_idx=0
)

tokenizer = SimpleTokenizer(vocab_size=10000)

# Generate code from description
description = "function to calculate fibonacci numbers"
# ... generation code here
```

## 📖 Usage

### Code → Description

Generate documentation and descriptions from code snippets:

```python
# Input code
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Generate description
description = model.generate_description(code)
print(description)
# Output: "recursive function to calculate fibonacci numbers"
```

### Description → Code

Generate code implementations from natural language descriptions:

```python
# Input description
description = "function to sort a list of numbers"

# Generate code
code = model.generate_code(description)
print(code)
# Output: "def sort_numbers(lst): return sorted(lst)"
```

## 🏋️ Training

### Local Training

1. **Prepare your data** in the `data/` directory
2. **Configure training parameters** in `train_transformer.py`
3. **Start training**:
   ```bash
   python train_transformer.py
   ```

### Docker Training

For containerized training, use the provided Docker setup:

```bash
# Build and run training container
docker-compose up codesage-train
```

## 🏗️ Model Architecture

CodeSage uses a Seq2Seq Transformer architecture with the following specifications:

- **Encoder**: 2 layers, 4 attention heads, 128 dimensions
- **Decoder**: 2 layers, 4 attention heads, 128 dimensions
- **Feed-forward**: 512 dimensions
- **Vocabulary**: 10,000 tokens
- **Parameters**: ~19M total

### Architecture Diagram

```
Input Code/Text → Tokenizer → Encoder → Decoder → Output Text/Code
     ↓              ↓          ↓         ↓           ↓
  Preprocessing → Embedding → Self-Attn → Cross-Attn → Generation
```

## 📚 API Reference

### Seq2SeqTransformer

The main model class located in `src/model/layers.py`.

#### Parameters

- `src_vocab_size` (int): Source vocabulary size
- `tgt_vocab_size` (int): Target vocabulary size  
- `d_model` (int): Model dimension (default: 128)
- `n_heads` (int): Number of attention heads (default: 4)
- `num_encoder_layers` (int): Number of encoder layers (default: 2)
- `num_decoder_layers` (int): Number of decoder layers (default: 2)
- `d_ff` (int): Feed-forward dimension (default: 512)
- `dropout` (float): Dropout rate (default: 0.1)
- `max_length` (int): Maximum sequence length (default: 64)
- `padding_idx` (int): Padding token index (default: 0)

#### Methods

- `forward(src, tgt)`: Forward pass
- `generate(src, max_new_tokens, bos_token_id, eos_token_id)`: Generate sequences
- `encode(src)`: Encode source sequence
- `decode(enc_src, tgt)`: Decode target sequence

### SimpleTokenizer

A basic tokenizer for demonstration purposes.

#### Methods

- `tokenize(text)`: Tokenize input text
- `encode(text, max_length)`: Encode text to token indices
- `decode(indices)`: Decode token indices to text

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to all functions and classes
- Write unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- Inspired by the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper
- Uses the [CodeSearchNet](https://github.com/github/CodeSearchNet) dataset
