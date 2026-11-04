import importlib.metadata

try:
    __version__ = importlib.metadata.version("llm_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from llm_basics.layers.layers import Linear
from llm_basics.layers.layers import Embedding
from llm_basics.layers.layers import RMSNorm
from llm_basics.layers.layers import SwiGLU
from llm_basics.layers.layers import RotaryPositionalEmbedding
from llm_basics.layers.layers import softmax
from llm_basics.layers.layers import MHSA
from llm_basics.layers.layers import TransformerBlock
from llm_basics.layers.layers import TransformerLM

from llm_basics.layers.moe_layers import SwitchTransformerLM