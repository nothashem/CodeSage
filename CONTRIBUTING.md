# Contributing to CodeSage 🤝

Thank you for your interest in contributing to CodeSage! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic knowledge of PyTorch and transformers

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/codesage.git
   cd codesage
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📝 Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for all function parameters and return values
- Keep line length under 88 characters (Black formatter default)
- Use meaningful variable and function names

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include examples in docstrings where appropriate
- Update README.md for new features

### Example Docstring

```python
def calculate_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Calculate cross-entropy loss between predictions and targets.
    
    Args:
        predictions: Model output logits of shape (batch_size, seq_len, vocab_size)
        targets: Target token indices of shape (batch_size, seq_len)
        
    Returns:
        Average loss across the batch
        
    Example:
        >>> pred = torch.randn(2, 10, 1000)
        >>> tgt = torch.randint(0, 1000, (2, 10))
        >>> loss = calculate_loss(pred, tgt)
    """
    return F.cross_entropy(predictions.view(-1, predictions.size(-1)), targets.view(-1))
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_model.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies
- Use fixtures for common test data

### Example Test

```python
import pytest
import torch
from src.model.layers import Seq2SeqTransformer

def test_transformer_forward():
    """Test that transformer forward pass works correctly."""
    model = Seq2SeqTransformer(
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        d_model=128,
        n_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2
    )
    
    src = torch.randint(0, 1000, (2, 10))
    tgt = torch.randint(0, 1000, (2, 8))
    
    output = model(src, tgt)
    assert output.shape == (2, 8, 1000)
```

## 🔄 Pull Request Process

### Before Submitting

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Run quality checks**
   ```bash
   # Format code
   black src/ tests/
   
   # Check for style issues
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Format

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Submitting the PR

1. Push your branch to your fork
2. Create a Pull Request on GitHub
3. Fill out the PR template
4. Request review from maintainers

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment information (OS, Python version, etc.)
- Error messages and stack traces

### Feature Requests

For feature requests, please include:

- Clear description of the feature
- Use cases and benefits
- Implementation suggestions (if any)
- Priority level

## 📚 Documentation

### Code Documentation

- All public APIs should be documented
- Include type hints
- Provide usage examples
- Document exceptions and edge cases

### User Documentation

- Update README.md for new features
- Add tutorials for complex features
- Include troubleshooting guides
- Keep installation instructions current

## 🏷️ Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Release notes written
- [ ] Tagged and pushed

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow the project's coding standards

### Communication

- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for questions and general discussion
- Be patient with responses
- Provide context when asking questions

## 🎯 Areas for Contribution

### High Priority

- Performance optimizations
- Better tokenization strategies
- Improved training data processing
- Enhanced evaluation metrics

### Medium Priority

- Additional model architectures
- Web interface
- API endpoints
- Docker containerization

### Low Priority

- Documentation improvements
- Code refactoring
- Test coverage improvements
- CI/CD enhancements

## 📞 Getting Help

- 📧 Email: your-email@example.com
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/codesage/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/codesage/issues)

Thank you for contributing to CodeSage! 🚀
