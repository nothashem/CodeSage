# CodeSage Repository Structure

This document explains the organized structure of the CodeSage GitHub repository.

## 📁 Directory Structure

```
codesage/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CHANGELOG.md                 # Version history and changes
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package installation script
├── 📄 pyproject.toml              # Modern Python project config
├── 📄 Makefile                    # Development commands
├── 📄 Dockerfile                  # Container configuration
├── 📄 docker-compose.yml          # Multi-service container setup
├── 📄 .dockerignore               # Docker build exclusions
├── 📄 .gitignore                  # Git exclusions
├── 📄 .pre-commit-config.yaml     # Code quality hooks
├── 📄 REPOSITORY_STRUCTURE.md     # This documentation file
│
├── 📁 src/                        # Source code package
│   ├── 📄 __init__.py            # Package initialization
│   └── 📁 model/                 # Model implementations
│       ├── 📄 __init__.py        # Model package init
│       └── 📄 layers.py          # Seq2SeqTransformer model
│
├── 📁 tests/                      # Test suite
│   ├── 📄 __init__.py            # Test package init
│   └── 📄 test_model.py          # Model tests
│
├── 📁 .github/                    # GitHub-specific files
│   ├── 📁 workflows/             # GitHub Actions CI/CD
│   │   └── 📄 ci.yml            # Continuous integration
│   ├── 📁 ISSUE_TEMPLATE/        # Issue templates
│   │   ├── 📄 bug_report.md     # Bug report template
│   │   └── 📄 feature_request.md # Feature request template
│   └── 📄 pull_request_template.md # PR template
│
├── 📁 data/                       # Data directory (gitignored)
│   ├── 📁 CodeSearchNet/         # Dataset files
│   ├── 📁 processed/             # Processed data
│   └── 📁 vocab/                 # Vocabulary files
│
├── 📁 checkpoints/               # Model checkpoints (gitignored)
│   ├── 📄 best_checkpoint.pt
│   ├── 📄 latest_checkpoint.pt
│   └── 📄 transformer_checkpoint.pt
│
├── 📄 interactive_ui.py          # Interactive CLI interface
├── 📄 train_transformer.py       # Training script
├── 📄 inference_script.py        # Inference script
└── 📄 research_paper.md          # Research documentation
```

## 🚀 Quick Start Guide

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/codesage.git
cd codesage
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Development Setup
```bash
# Install development dependencies
make install-dev

# Or manually:
pip install -e ".[dev]"
pre-commit install
```

### 3. Run the Application
```bash
# Interactive CLI
python interactive_ui.py

# Or using the installed command
codesage
```

### 4. Development Workflow
```bash
# Run tests
make test

# Check code quality
make quality

# Format code
make format

# Full development cycle
make full-test
```

## 🐳 Docker Usage

### Development Environment
```bash
# Build and run development container
docker-compose up codesage-dev

# Or build manually
docker build -t codesage .
docker run -it --rm -v $(pwd):/app codesage /bin/bash
```

### Production
```bash
# Run the application
docker-compose up codesage

# Train the model (with GPU support)
docker-compose up codesage-train
```

## 🔧 Key Files Explained

### Core Application Files
- **`interactive_ui.py`**: Main CLI interface for user interaction
- **`train_transformer.py`**: Training script for the model
- **`src/model/layers.py`**: Core Seq2SeqTransformer implementation

### Configuration Files
- **`pyproject.toml`**: Modern Python project configuration
- **`setup.py`**: Package installation and metadata
- **`requirements.txt`**: Python dependencies
- **`.pre-commit-config.yaml`**: Code quality automation

### Documentation
- **`README.md`**: Comprehensive project overview
- **`CONTRIBUTING.md`**: Contribution guidelines
- **`CHANGELOG.md`**: Version history
- **`REPOSITORY_STRUCTURE.md`**: This structure documentation

### Development Tools
- **`Makefile`**: Convenient development commands
- **`Dockerfile`**: Container configuration
- **`docker-compose.yml`**: Multi-service setup
- **`.github/workflows/ci.yml`**: Automated testing

### GitHub Templates
- **`.github/ISSUE_TEMPLATE/bug_report.md`**: Standardized bug reports
- **`.github/ISSUE_TEMPLATE/feature_request.md`**: Feature request template
- **`.github/pull_request_template.md`**: PR submission template

## 🧪 Testing

### Run Tests
```bash
# Basic tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_model.py -v
```

### Code Quality
```bash
# Linting
make lint

# Type checking
make type-check

# Security checks
make security

# All quality checks
make quality
```

## 📦 Package Management

### Install as Package
```bash
# Development install
pip install -e .

# Production install
pip install .
```

### Build Distribution
```bash
# Build wheel and source distribution
make build

# Or manually
python -m build
```

## 🔄 CI/CD Pipeline

The repository includes a comprehensive CI/CD pipeline:

1. **Automated Testing**: Runs on Python 3.8-3.11
2. **Code Quality**: Linting, formatting, type checking
3. **Security**: Bandit security analysis
4. **Build**: Package building and artifact upload

### GitHub Actions Workflow
- Triggers on push to main/develop and pull requests
- Matrix testing across Python versions
- Automated coverage reporting
- Build artifact generation

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `make install-dev`
4. Make your changes
5. Run quality checks: `make quality`
6. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Use type hints
- Write comprehensive tests
- Update documentation
- Use conventional commit messages

### Pre-commit Hooks
The repository includes pre-commit hooks that automatically:
- Format code with Black
- Sort imports with isort
- Check code style with flake8
- Run type checking with mypy
- Perform security analysis with bandit

## 📚 Documentation

### API Documentation
- Model architecture in `src/model/layers.py`
- Interactive interface in `interactive_ui.py`
- Training configuration in `train_transformer.py`

### User Guides
- Quick start in `README.md`
- Repository structure in `REPOSITORY_STRUCTURE.md`

### Development Documentation
- Contributing guidelines in `CONTRIBUTING.md`
- Version history in `CHANGELOG.md`
- Development commands in `Makefile`

## 🎯 Project Organization Benefits

### Professional Structure
- **Standard Python Package Layout**: Follows Python packaging best practices
- **Comprehensive Documentation**: Multiple levels of documentation for different audiences
- **Automated Quality Assurance**: Pre-commit hooks and CI/CD pipeline
- **Container Support**: Docker and docker-compose for easy deployment

### Developer Experience
- **Easy Setup**: Simple installation and development environment setup
- **Quality Tools**: Automated formatting, linting, and testing
- **Clear Guidelines**: Well-defined contribution process and coding standards
- **Comprehensive Testing**: Unit tests with coverage reporting

### Scalability
- **Modular Design**: Clean separation of concerns with proper package structure
- **Extensible Architecture**: Easy to add new features and models
- **Container Ready**: Docker support for easy deployment
- **Community Friendly**: Templates and guidelines for contributors

## 🎯 Next Steps

1. **Customize**: Update personal information in configuration files
2. **Deploy**: Set up GitHub repository and enable Actions
3. **Document**: Add project-specific documentation
4. **Extend**: Add new features and improvements
5. **Community**: Engage with contributors and users

## 📞 Support

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the README and contributing guidelines
- **Email**: Contact maintainers for direct support

---

This structure provides a professional, maintainable, and scalable foundation for the CodeSage project. All files are organized according to Python and GitHub best practices, making it easy for contributors to understand and work with the codebase. 