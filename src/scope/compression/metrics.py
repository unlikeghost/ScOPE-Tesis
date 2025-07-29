# -*- coding: utf-8 -*-

from enum import Enum
from abc import ABC, abstractmethod
from typing import Union

from scope.compression.compressors import get_compressor, _BaseCompressor


class MetricType(Enum):
    NCD = "ncd"
    # NCCD = "nccd" -> This is not use bc this method is not computable
    CDM = "cdm"
    NRC = "nrc"
    CLM = "clm"
    MSE = "mse"
    RMSE = "rmse"


class _BaseMetric(ABC):
    """Estrategia abstracta para métricas"""
    
    def __init__(self, compressor_name: str, concat_value: str = " ", compression_level: int = 9, min_size_threshold: int = 0):
        self._compressor: _BaseCompressor = get_compressor(
            name=compressor_name,
            compression_level=compression_level,
            min_size_threshold=min_size_threshold
        )
        
        self._compressor_name = compressor_name
        self._compression_level = compression_level,
        self._min_size_threshold = min_size_threshold
        self._concat_value = concat_value
    
    def __repr__(self) -> str:
        return (
            f"CompressionMetric("
                f"Metric='{self.name}', "
                f"Compressor='{self._compressor_name}', "
                f"Compression Level={self._compression_level}, "
                f"Min Size Threshold={self._min_size_threshold}"
            f")"
    )
    
    def __str__(self):
        return self.__repr__()
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> float:
        raise NotImplementedError("")
    
    @staticmethod
    def concat_str(x1: str, x2: str, concat_str: str = '') -> str:
        if not all(isinstance(sequence, str) for sequence in (x1, x2)):
            raise ValueError("Inputs to _join must be strings.")  
          
        return f"{x1}{concat_str}{x2}"
    
    def get_compress_len(self, sequence) -> int:
        _, _, len_compression = self._compressor(sequence)
        return len_compression


class NCDMetric(_BaseMetric):
    name = "NCD"
    
    def compute(self, x1: str, x2: str, concat_str: str = '') -> float:
        
        x1x2 = self.concat_str(x1=x1, x2=x2, concat_str=concat_str)
        
        c_x1 = self.get_compress_len(x1)
        c_x2 = self.get_compress_len(x2)
        c_x1x2 = self.get_compress_len(x1x2)
        
        denom = max(c_x1, c_x2)
        if denom == 0:
            raise ZeroDivisionError("Denominator in NCD is zero.")
        
        return (c_x1x2 - min(c_x1, c_x2)) / denom


class NCCDMetric(_BaseMetric):
    name = "NCCD"
    
    def compute(self, x1: str, x2: str, concat_str: str = '') -> float:
        x1x2 = self.concat_str(x1=x1, x2=x2, concat_str=concat_str)
        x2x1 = self.concat_str(x1=x2, x2=x1, concat_str=concat_str)
        
        c_x1 = self.get_compress_len(x1)
        c_x2 = self.get_compress_len(x2)
        
        c_x1x2 = self.get_compress_len(x1x2)
        c_x2x1 = self.get_compress_len(x2x1)
        
        denom = max(c_x1, c_x2)
        
        if denom == 0:
            raise ZeroDivisionError("Denominator in NCCD is zero.")
        
        numer = max(c_x1x2, c_x2x1)
        
        return numer / denom


class CDMMetric(_BaseMetric):
    name = "CDM"
    
    def compute(self, x1: str, x2: str, concat_str: str = '') -> float:
        x1x2 = self.concat_str(x1=x1, x2=x2, concat_str=concat_str)
        c_x1 = self.get_compress_len(x1)
        c_x2 = self.get_compress_len(x2)
        c_x1x2 = self.get_compress_len(x1x2)
        
        denom = c_x1 + c_x2
        if denom == 0:
            raise ZeroDivisionError("Denominator in CDM is zero.")
        
        return c_x1x2 / denom


class NRCMetric(_BaseMetric):
    name = "NRC"
    
    def compute(self, x1: str, x2: str, concat_str: str = '') -> float:
        x1x2 = self.concat_str(x1=x1, x2=x2, concat_str=concat_str)
        
        c_x1 = self.get_compress_len(x1)
        c_x1x2 = self.get_compress_len(x1x2)
        
        if c_x1 == 0:
            raise ZeroDivisionError("Denominator in NRC is zero.")
        
        return c_x1x2 / c_x1


class CLMMetric(_BaseMetric):
    name = "CLM"
    
    def compute(self, x1: str, x2: str, concat_str: str = '') -> float:
        x1x2 = self.concat_str(x1=x1, x2=x2, concat_str=concat_str)

        c_x1 = self.get_compress_len(x1)
        c_x2 = self.get_compress_len(x2)
        c_x1x2 = self.get_compress_len(x1x2)
        
        
        denominator: float = c_x1x2
        if c_x1==0:
            raise ZeroDivisionError("Denominator in clm is zero.")
        
        numerator: float = c_x1 + c_x2 - c_x1x2
        
        return 1 - (numerator / denominator)
    


class MSEMetric(_BaseMetric):
    name = "MSE"
    
    def compute(self, x1: str, x2: str, **kw) -> float:
        c_x1 = self.get_compress_len(x1)
        c_x2 = self.get_compress_len(x2)
        
        if c_x1 is None or c_x2 is None:
            raise ValueError("Compressed lengths cannot be None")
        
        squared_error = (c_x1 - c_x2) ** 2
        
        return float(squared_error)


class RMSEMetric(_BaseMetric):
    name = "RMSE"
    
    def compute(self, x1: str, x2: str, **kw) -> float:
        c_x1 = self.get_compress_len(x1)
        c_x2 = self.get_compress_len(x2)
        
        if c_x1 is None or c_x2 is None:
            raise ValueError("Compressed lengths cannot be None")
        
        squared_error = (c_x1 - c_x2) ** 2
        rmse = squared_error ** 0.5
        
        return float(rmse)


METRIC_STRATEGIES = {
    MetricType.NCD: NCDMetric,
    # MetricType.NCCD: NCCDMetric,
    MetricType.NRC: NRCMetric,
    MetricType.CDM: CDMMetric,
    MetricType.CLM: CLMMetric,
    MetricType.MSE: MSEMetric,
    MetricType.RMSE: RMSEMetric
}


def get_metric(
    name: Union[str, MetricType],
    compressor_name: str = "gzip",
    compression_level: int = 9,
    min_size_threshold: int = 0,
    concat_value: str = " "
) -> _BaseMetric:
    if isinstance(name, str):
        try:
            metric_enum = MetricType(name.lower())
        except ValueError:
            allowed = sorted(m.value for m in MetricType)
            raise ValueError(
                f"'{name}' is not a valid metric name. "
                f"Expected one of: {', '.join(allowed)}"
            )
    elif isinstance(name, MetricType):
        metric_enum = name
    else:
        raise TypeError("Expected 'name' to be str or MetricType.")
    
    metric_class = METRIC_STRATEGIES[metric_enum]
    return metric_class(
        compressor_name=compressor_name,
        compression_level=compression_level,
        min_size_threshold=min_size_threshold,
        concat_value=concat_value
    )


if __name__ == '__main__':
    
    for metric in MetricType:
        this_metric = METRIC_STRATEGIES[metric](
            compressor_name="gzip"
        )
        
        print(this_metric)
        print(this_metric.compute(x1='this is a test', x2='this is a test', concat_str=''))
        
    print("="*30)
    
    metric = get_metric(
        name="ncd",
        compressor_name="lzma",
        compression_level=7,
        min_size_threshold=100,
        concat_value=""
    )

    print(metric)
    result = metric.compute("hello world", "hello world")
    print(result)