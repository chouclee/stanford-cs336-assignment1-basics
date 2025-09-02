import importlib.metadata

from .bpe import train_bpe
from .tokenizer import Tokenizer
from .linear import Linear

__version__ = importlib.metadata.version("cs336_basics")

