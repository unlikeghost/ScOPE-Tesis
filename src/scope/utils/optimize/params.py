import warnings
from enum import Enum
from typing import List
from dataclasses import dataclass, field

from scope.compression.metrics import MetricType
from scope.compression.compressors import CompressorType
from scope.distances import MatchingType

warnings.filterwarnings('ignore')


class OptimizationDirection(Enum):
    """Direction for optimization objectives."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass
class ObjectiveConfig:
    """Configuration for a single optimization objective."""
    name: str
    direction: OptimizationDirection
    weight: float = 1.0
    
    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError("Weight must be positive")


@dataclass 
class ParameterSpace:
    """Define the parameter space for the ScOPE model optimization."""

    # Categorical parameters
    compressor_names: List[str] = field(
        default_factory=lambda: [compressor.value for compressor in CompressorType]
    )
    
    compression_metrics: List[str] = field(
        default_factory=lambda: [metric.value for metric in MetricType]
    )
    
    string_separators: List[str] = field(
        default_factory=lambda: [' ', '\t', '\n', '|', ',']
    )
    
    model_types: List[str] = field(
        default_factory=lambda: ["ot", "pd"]
    )

    # Integer parameters
    compression_levels: List[int] = field(
        default_factory=lambda: [1, 2, 5, 7, 9]
    )
    
    min_size_thresholds: List[int] = field(
        default_factory=lambda: [0, 10, 50]
    )

    # Boolean parameters
    use_best_sigma_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    symetric_matrix_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )

    # Parameters specific to ScOPEOT
    ot_use_matching_method_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    ot_matching_method_names: List[str] = field(
        default_factory=lambda: [matching.value for matching in MatchingType]
    )
    
    # Parameters specific to ScOPEPD
    pd_distance_metrics: List[str] = field(
        default_factory=lambda: ["cosine", "euclidean", "manhattan", "minkowski", "dot_product"]
    )
    
    pd_use_prototypes_options: List[bool] = field(
        default_factory=lambda: [True, False]
    )
    
    def get_total_combinations(self) -> int:
        """Calculate total possible parameter combinations."""
        total_basic = (
            len(self.compressor_names) * 
            len(self.compression_metrics) * 
            len(self.compression_levels) * 
            len(self.min_size_thresholds) *  
            len(self.string_separators) * 
            len(self.use_best_sigma_options) *
            len(self.symetric_matrix_options)
        )
        
        total_ot = (
            len(self.ot_use_matching_method_options) * 
            max(1, len(self.ot_matching_method_names))
        )
        
        total_pd = (
            len(self.pd_distance_metrics) * 
            len(self.pd_use_prototypes_options)
        )
        
        return total_basic * (total_ot + total_pd)
    
    def validate(self) -> None:
        """Validate parameter space configuration."""
        if not self.compressor_names:
            raise ValueError("At least one compressor must be specified")
        if not self.compression_metrics:
            raise ValueError("At least one compression metric must be specified")
        if not self.model_types:
            raise ValueError("At least one model type must be specified")

