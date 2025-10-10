"""
Neural network models for character-level language modeling.
"""

from .bigram import BigramModel
from .mlp import MLP

__all__ = ["BigramModel", "MLP"]
