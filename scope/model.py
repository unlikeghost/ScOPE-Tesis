from typing import Dict, List, Union, Any, Generator

from scope.compressors import get_compressor
from scope.matrix import MatrixFactory as MatrixFactory
from scope.models import ModelRegistry
    

class ScOPE:
    
    def __init__(self,
                 compressor_name: str,
                 compression_distance_function: str,
                 use_best_sigma: bool = True,
                 string_separator: str = ' ',
                 model_type: str = "ot",
                 get_softmax: bool = True,
                 **model_kwargs) -> None:
                
        _compressor = get_compressor(compressor_name)
        self.matrix_factory = MatrixFactory(
            compressor_module=_compressor,
            name_distance_function=compression_distance_function,
            str_separator=string_separator
        )
        
        self.model = ModelRegistry.create(model_type, **model_kwargs)
        self.use_best_sigma = use_best_sigma
        self.model_type = model_type
        
        self.get_softmax = get_softmax
    
    def __str__(self):
        return (f"ScOPE Model: {self.model_type}, "
                f"Use Best Sigma: {self.use_best_sigma}, "
                f"Model Parameters: {self.model}")
    
    def __repr__(self):
        return self.__str__()
    
    
    def __forward__(self, sample: str, kw_samples: Dict[str, str]) -> Dict[str, Any]:
        
        matrix_result: dict = self.matrix_factory(sample, kw_samples, get_best_sigma=self.use_best_sigma)
        
        predictions: dict = self.model(matrix_result, softmax=self.get_softmax)[0]
        
        return predictions

    def forward(self, list_samples: List[str], list_kw_samples: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        for index, (sample, kw_samples) in enumerate(zip(list_samples, list_kw_samples)):
            prediction = self.__forward__(sample, kw_samples)
            yield prediction
    
    def __call__(self, 
                 list_samples: Union[List[str], str],
                 list_kw_samples: Union[
                     List[Dict[str, str]],
                     Dict[str, str]]
                     ) -> Generator[Dict[str, float], None, None]:
        
        if not isinstance(list_samples, list):
            list_samples = [list_samples]
        
        if not isinstance(list_kw_samples, list):
            list_kw_samples = [list_kw_samples]
        
        
        if len(list_samples) != len(list_kw_samples):
            raise ValueError(
                "The number of samples and keyword samples must be the same."
            )
        
        return self.forward(
            list_kw_samples=list_kw_samples,
            list_samples=list_samples
        )
