import os
import pickle
import warnings
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any, Union

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from scope.model import ScOPE
from scope.utils.optimize.base import ScOPEOptimizer
from scope.utils.optimize.params import ParameterSpace, ObjectiveConfig, OptimizationDirection

warnings.filterwarnings('ignore')


class ScOPEOptimizerBayesian(ScOPEOptimizer):
    
    def __init__(self, 
                parameter_space: Optional[ParameterSpace] = None,
                free_cpu: int = 0,
                n_trials: int = 50,
                timeout: int = 1800,
                random_seed: int = 42,
                cv_folds: int = 3,
                target_metric: Union[str, Dict[str, float]] = 'auc_roc',
                study_name: str = "scope_optimization",
                output_path: str = "./results",
                sampler_config: Optional[Dict[str, Any]] = None,
                pruner_config: Optional[Dict[str, Any]] = None,
                use_cache: bool = True,
                fail_fast: bool = False
    ):
        
        super().__init__(
            parameter_space=parameter_space,
            objectives=processed_objectives,
            free_cpu=free_cpu,
            random_seed=random_seed,
            cv_folds=cv_folds,
            fail_fast=fail_fast
        )
    