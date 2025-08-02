# -*- coding: utf-8 -*-

import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Any, List, Dict, Union

from scope.compression.metrics import get_metric, _BaseMetric


class _BaseMatrixFactory(ABC):
    
    def __init__(self,
                 compression_metric: str = "ncd",
                 concat_value: str = " ",
                 compressor_name: str = "gzip",
                 compression_level: int = 9,
                 symetric: bool = False,
                 min_size_threshold: int = 0):
        
        if not isinstance(concat_value, str):
            raise ValueError("`concat_value` must be a string.")
           
        self._concat_value = concat_value
        
        self._compresion_metric: _BaseMetric = get_metric(
            name=compression_metric,
            compressor_name=compressor_name,
            compression_level=compression_level,
            min_size_threshold=min_size_threshold,
            concat_value=concat_value,
        )
        
        self._compression_metric_name = compression_metric
        
        self._is_symetric: bool = symetric
        
    def __str__(self) -> str:
        return (
                f"CompressionMatrixGenerator("
                    f"Compressor='{self._compresion_metric._compressor_name}', "
                    f"Compression Level={self._compresion_metric._compression_level}, "
                    f"Min Size Threshold={self._compresion_metric._min_size_threshold}, "
                    f"String Concatenation Value='{repr(self._concat_value)}'"
                    f"Compression Metric={self._compression_metric_name}"
                f")"
            )
    
    def __repr__(self):
        return self.__str__()
        
    
    def get_best_sigma(self, sample: str, *kw_samples: str) -> float:
        """
        Computes the average sigma for the provided data.

        Parameters:
        sample: str
            Base sample for comparison.
        *kw_samples: str
            Additional samples.

        Returns:
        float
            The average sigma.
        """
        all_data = np.array([sample] + list(kw_samples), dtype=object)
        
        def compute_sigma(sequence: str) -> float:
            """Computes the sigma for a specific sequence."""
            return self._compresion_metric.compute(x1=sequence, x2=sequence, concat_str=self._concat_value)
            
        compute_sigma_vec = np.vectorize(compute_sigma)
        sigmas = compute_sigma_vec(all_data)
        
        return np.mean(sigmas)

    @abstractmethod
    def compute_compression_matrix(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def build_matrix(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    
    def __call__(self, *args, **kwargs) -> Any:
        return self.build_matrix(*args, **kwargs)



class CompressionMatrixFactory(_BaseMatrixFactory):
    
    def __init__(self, compression_metric = "ncd", concat_value = " ",
                 compressor_name = "gzip", compression_level = 9,
                 min_size_threshold = 0, symetric: bool = False):
                
        super().__init__(compression_metric, concat_value,
                        compressor_name, compression_level,
                        symetric, min_size_threshold)
        
        
    def compute_compression_matrix(self, samples: List[str]) -> np.ndarray:
        sample_count = len(samples)
        
        distance_matrix = np.zeros(shape=(sample_count, sample_count), dtype=np.float32)
        
        for i in range(sample_count):
            for j in range(i, sample_count):
                
                x1: str = samples[i]
                x2: str = samples[j]
                
                x1x2_distance_value: float = self._compresion_metric.compute(
                        x1=x1,
                        x2=x2,
                        concat_str=self._concat_value
                    )
                                    
                if self._is_symetric:
                    distance_matrix[i, j] = distance_matrix[j, i] = x1x2_distance_value
                    
                else:
                    x2x1_distance_value: float = self._compresion_metric.compute(
                        x1=x2,
                        x2=x1,
                        concat_str=self._concat_value
                    )

                    distance_matrix[i, j] = x1x2_distance_value
                    distance_matrix[j, i] = x2x1_distance_value
                    
        return distance_matrix
                
    def build_matrix(self,  
                     sample: str,
                     kw_samples: Union[Dict[Union[int, str], str], Tuple, List],
                     get_sigma: bool = False) -> Dict[str, np.ndarray]:
        
        if isinstance(kw_samples, dict):
            cluster_samples = copy.deepcopy(kw_samples)
        
        else:
            cluster_samples = {
                index: value for index, value in enumerate(kw_samples)
            }
        
        results = {}
        all_sigmas = []
                
        for cluster_key, cluster_values in cluster_samples.items():
            
            cluster_values += [sample]
            
            compression_matrix = self.compute_compression_matrix(cluster_values)
            
            # not sure how to return dict form!!!
            
            # results[cluster_key] = {
            #     'ScOPE_KwSample': compression_matrix[:-1, :],
            #     'ScOPE_Sample': compression_matrix[-1:, :]
            # }
            
            results[f"ScOPE_KwSamples_{cluster_key}"] = compression_matrix[:-1, :]
            results[f"ScOPE_UkSample_{cluster_key}"] = compression_matrix[-1:, :]
            
            if get_sigma:
                sigma = self.get_best_sigma(sample, *cluster_values[:-1])
                all_sigmas.append(sigma)
        
        if get_sigma:
            results["sigma"] = np.mean(all_sigmas)
        
        return results
            

if __name__ == "__main__":
    from scope.compression.matrix import CompressionMatrixFactory

    test_samples = {
        'class_0': ['Hola', 'Adios', 'Buenos dias'],
        'class_1': ['Hello', 'Goodbye', 'Good morning']
    }
    test_sample = 'Hello'
       
    matrix = CompressionMatrixFactory(
        concat_value="",
        min_size_threshold=0
    )
   
    result = matrix.build_matrix(
        sample=test_sample,
        kw_samples=test_samples,
        get_sigma=True
    )
    
    print(result)