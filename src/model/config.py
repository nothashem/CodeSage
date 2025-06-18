from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    # Model dimensions
    vocab_size: int
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_length: int = 512
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # Generation
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    
    # Optional features
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate configuration values."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.d_ff > self.d_model, "d_ff should be larger than d_model"
        assert 0 <= self.dropout < 1, "dropout must be between 0 and 1"
        assert 0 <= self.attention_dropout < 1, "attention_dropout must be between 0 and 1"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.weight_decay >= 0, "weight_decay must be non-negative"
        assert 0 < self.temperature <= 1, "temperature must be between 0 and 1"
        assert self.top_k > 0, "top_k must be positive"
        assert 0 < self.top_p <= 1, "top_p must be between 0 and 1" 