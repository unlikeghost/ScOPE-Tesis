from .ScopeOT import ScOPEOT
from .ScopePD import ScOPEPD
from .base_model import BaseModel

from typing import List, Dict, Any, Type


class ModelRegistry:
    """Simple registry para manejar modelos disponibles"""
    
    _models: Dict[str, Type[BaseModel]] = {}
    _defaults: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel], defaults: Dict[str, Any] = None):
        cls._models[name] = model_class
        cls._defaults[name] = defaults or {}
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        config = cls._defaults[name].copy()
        config.update(kwargs)
        return cls._models[name](**config)
    
    @classmethod
    def available(cls) -> List[str]:
        return list(cls._models.keys())
    

ModelRegistry.register(
    name="ot",
    model_class=ScOPEOT,
    defaults={
        "use_matching_method": True,
        "matching_method_name": "dice"
    }
)

ModelRegistry.register(
    name="pd",
    model_class=ScOPEPD,
    defaults={
        "distance_metric": "cosine"
    }
)


__all__ = ["ScOPEOT", "ScOPEPD", "ModelRegistry"]
