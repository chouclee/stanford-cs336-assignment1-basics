import importlib.metadata

from .bpe import train_bpe
from .tokenizer import Tokenizer

__version__ = importlib.metadata.version("cs336_basics")

