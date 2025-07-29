from scope.predictor.ot import ScOPEOT
from scope.predictor.pd import ScOPEPD
from scope.predictor.base import _BasePredictor

from typing import List, Dict, Any, Type


class PredictorRegistry:
    
    _predictors: Dict[str, Type[_BasePredictor]] = {}
    _defaults: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[_BasePredictor],defaults: Dict[str, Any] = None):
        cls._predictors[name] = model_class
        cls._defaults[name] = defaults or {}
    
    @classmethod
    def create(cls, name: str, use_softmax: bool = True, epsilon: float = 1e-8, **kwargs) -> _BasePredictor:
        if name not in cls._predictors:
            available = list(cls._predictors.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        config = cls._defaults[name].copy()
        
        config['use_softmax'] = use_softmax
        config['epsilon'] = epsilon
        
        config.update(kwargs)
        return cls._predictors[name](**config)
    
    @classmethod
    def available(cls) -> List[str]:
        return list(cls._predictors.keys())
    

PredictorRegistry.register(
    name="ot",
    model_class=ScOPEOT,
    defaults={
        "use_matching_method": True,
        "matching_method_name": "dice",
        "use_softmax": True,
        "epsilon": 1e-8
    }
)

PredictorRegistry.register(
    name="pd",
    model_class=ScOPEPD,
    defaults={
        "distance_metric": "cosine",
        "use_prototypes": False,
        "use_softmax": True,
        "epsilon": 1e-8
    }
)
