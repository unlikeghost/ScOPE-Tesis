
from typing import List, Union, Dict, Generator

from scope.models import ScOPEOT
from scope.compressors import get_compressor
from scope.matrix import MatrixFactory as MatrixFactory


class ScOPE:
    def __init__(self,
                 compressor: str,
                 name_distance_function: str,
                 use_best_sigma: bool = True,
                 str_separator: str = ' ',
                 use_matching_method: bool = True
                 ) -> None:
        
        _compressor = get_compressor(compressor)

        self.matrix_factory: MatrixFactory = MatrixFactory(
            compressor_module=_compressor,
            name_distance_function=name_distance_function,
            str_separator=str_separator
        )
        
        self.model = ScOPEOT(
            use_matching_method=use_matching_method
        )
        
        self.use_best_sigma = use_best_sigma
    
    def __forward__(self, sample: str, kw_samples: Dict[str, str]) -> Dict[str, float]:        
        matrix_result: dict = self.matrix_factory(sample, kw_samples, get_best_sigma=self.use_best_sigma)
        predictions: dict = self.model.forward(matrix_result, softmax=True)
        return predictions['softmax']
    
    def forward(self, list_samples: List[str], list_kw_samples: List[Dict[str, str]]) -> Generator[Dict[str, float], None, None]:
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
    
    
if __name__ == '__main__':
    
    model =  ScOPE(
        compressor="gzip",
        name_distance_function='ncd',
        use_best_sigma=True,
        str_separator=' ',
        use_matching_method=True
    )
    
    sample= ["hola"]
    
    kw_samples = {
        "sample_1": ["hola"],
        "sample_2": ["adios"]
    }
        
    print(list(model(sample, kw_samples)))