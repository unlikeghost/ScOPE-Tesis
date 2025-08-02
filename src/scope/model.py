from typing import Dict, List, Union, Any, Generator

from scope.compression import CompressionMatrixFactory
from scope.predictor import PredictorRegistry


class ScOPE:
    
    def __init__(self,
                 compressor_name: str,
                 compression_metric: str,
                 compression_level: int = 9,
                 min_size_threshold: int = 0,
                 use_best_sigma: bool = True,
                 string_separator: str = ' ',
                 use_softmax: bool = True,
                 symetric_matrix: bool = False,
                 model_type: str = "ot",
                 **model_kwargs) -> None:
        
        self.matrix_generator = CompressionMatrixFactory(
            compression_metric=compression_metric,
            compressor_name=compressor_name,
            compression_level=compression_level,
            min_size_threshold=min_size_threshold,
            concat_value=string_separator,
            symetric=symetric_matrix,
        )
        
        self.predictor = PredictorRegistry.create(
            name=model_type,
            use_softmax=use_softmax,
            **model_kwargs
        )
        
        self._symetric_matrix = symetric_matrix
        self._compressor_name = compressor_name
        self._compression_metric = compression_metric
        self._compression_level = compression_level
        self._min_size_threshold = min_size_threshold
        self._string_separator = string_separator
        self._using_sigma = use_best_sigma
        self._use_softmax = use_softmax
        self._model_type = model_type
        self._model_kwargs = model_kwargs
    
    def __forward__(self, sample: str, kw_samples: Dict[str, str]) -> Dict[str, Any]:
        
        matrix_result: dict = self.matrix_generator(sample, kw_samples, get_sigma=self._using_sigma)
        
        predictions: dict = self.predictor(matrix_result)[0]
        
        return predictions

    def forward(self, list_samples: List[str], list_kw_samples: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        
        for sample, kw_samples in zip(list_samples, list_kw_samples):
            prediction = self.__forward__(sample, kw_samples)
            
            yield prediction

    def __call__(self, 
                 list_samples: Union[List[str], str],
                 list_kw_samples: Union[
                     List[
                         Dict[str, str]
                        ],
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
    
    def __str__(self):
        return (f"ScOPE(compressor='{self._compressor_name}', "
                f"Symetric Matrix = '{self._symetric_matrix}', "
                f"metric='{self._compression_metric}', "
                f"level={self._compression_level}, "
                f"min_size={self._min_size_threshold}, "
                f"use_sigma={self._using_sigma}, "
                f"softmax={self._use_softmax}, "
                f"model='{self._model_type}',"
                f"model_args='{self._model_kwargs}')")

    def __repr__(self):
        return self.__str__()