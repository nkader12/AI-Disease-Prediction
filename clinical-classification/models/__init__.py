"""
Model implementations for clinical classification.

This package contains:
- BaselineEnsemble: Traditional ML ensemble with regex, logistic regression,
  random forest, and semi-supervised learning
- LLMAgentSystem: Three-agent LLM system using OpenAI GPT-4 and Astra DB
"""

from .baseline import BaselineEnsemble
from .experimental import LLMAgentSystem

__all__ = [
    'BaselineEnsemble',
    'LLMAgentSystem',
]
