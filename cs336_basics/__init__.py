import importlib.metadata

from .bpe import train_bpe
from .tokenizer import Tokenizer
from .linear import Linear
from .embedding import Embedding

__version__ = importlib.metadata.version("cs336_basics")

