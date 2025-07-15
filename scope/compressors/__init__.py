from .gzip_compressor import GZIPCompressor
from .bz2_compressor import BZ2Compressor
from .lz77_compressor import LZ77Compressor
from .base_compressor import BaseCompressor
from .zstandard_compressor import ZStandardCompressor

# Mapping of string names to compressor classes
COMPRESSOR_REGISTRY = {
    'gzip': GZIPCompressor,
    'bz2': BZ2Compressor,
    'lz77': LZ77Compressor,
    'zstandard': ZStandardCompressor,
}

def get_compressor(name: str):
    """
    Factory function to get a compressor instance by string name.
    
    Args:
        name (str): The name of the compressor ('gzip', 'bz2', 'lz77', 'zstandard', 'base')
        
    Returns:
        BaseCompressor: An instance of the requested compressor
        
    Raises:
        ValueError: If the compressor name is not recognized
        
    Example:
        >>> compressor = get_compressor('gzip')
        >>> result = compressor('Hello World')
    """
    if name.lower() not in COMPRESSOR_REGISTRY:
        available = ', '.join(COMPRESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown compressor '{name}'. Available compressors: {available}")
    
    return COMPRESSOR_REGISTRY[name.lower()]()

__all__ = [
    'BaseCompressor',
    'get_compressor'
]
