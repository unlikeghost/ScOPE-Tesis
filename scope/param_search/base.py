import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold

from dataclasses import dataclass, field

from scope import ScOPE
from scope.utils.report_generation import make_report


warnings.filterwarnings('ignore')


@dataclass
class ParameterSpace:
    """Define the parameter space for the ScOPE model optimization."""

    # Categorical parameters
    compressor_names: List[str] = field(default_factory=lambda: ['gzip', 'bz2', 'lz77', 'zstandard'])
    compression_distance_functions: List[str] = field(default_factory=lambda: ['ncd', 'cdm', 'clm', 'mse'])
    str_separators: List[str] = field(default_factory=lambda: [' ', '\t', '\n', '|', ','])
    model_types: List[str] = field(default_factory=lambda: ["ot", "pd"])

    # Boolean parameters
    use_best_sigma_options: List[bool] = field(default_factory=lambda: [True, False])

    # Parameters specific to ScOPEOT
    ot_use_matching_method_options: List[bool] = field(default_factory=lambda: [True, False])
    ot_matching_method_names: List[Optional[str]] = field(default_factory=lambda: [None, "matching", "jaccard", "dice", "overlap"])

    # Parameters specific to ScOPEPD
    pd_distance_metrics: List[str] = field(default_factory=lambda: ["cosine", "euclidean", "manhattan", "chebyshev", "canberra", "minkowski", "braycurtis", "hamming", "correlation", "dot_product"])
    pd_use_prototypes_options: List[bool] = field(default_factory=lambda: [True, False])

    # Continuous parameters
    # epsilon_range: Tuple[float, float] = (1e-8, 1e-4)


class ScOPEOptimizer(ABC):
    """
        Abstract base class for ScOPE model optimizers.
    """

    def __init__(self, parameter_space: ParameterSpace = None):
        self.parameter_space = parameter_space or ParameterSpace()
    
    def create_model_from_params(self, params: Dict[str, Any]) -> ScOPE:
        """create a ScOPE model instance from the given parameters."""

        # Common base parameters
        base_params = {
            'compressor_name': params['compressor_name'],
            'compression_distance_function': params['compression_distance_function'],
            'use_best_sigma': params['use_best_sigma'],
            'string_separator': params['str_separator'],
            'model_type': params['model_type'],
            'use_softmax': True
        }

        # Model-specific parameters based on type
        model_kwargs = {}
        
        if params['model_type'] == "ot":
            # Parameters for ScOPEOT
            model_kwargs['use_matching_method'] = params.get('ot_use_matching_method', False)
            if params.get('ot_matching_method_name') is not None:
                model_kwargs['matching_method_name'] = params['ot_matching_method_name']
                
        elif params['model_type'] == "pd":
            # Parameters for ScOPEPD
            model_kwargs['distance_metric'] = params.get('pd_distance_metric', 'cosine')
            model_kwargs['use_prototypes'] = params.get('pd_use_prototypes', False)

        # Combine base parameters with model-specific ones
        all_params = {**base_params, **model_kwargs}
        
        return ScOPE(**all_params)
    
    @abstractmethod
    def suggest_categorical_params(self, trial) -> Dict[str, Any]:
        """Suggest categorical parameters"""
        pass
    
    @abstractmethod
    def suggest_boolean_params(self, trial) -> Dict[str, Any]:
        """Suggest boolean parameters"""
        pass
    
    @abstractmethod
    def suggest_continuous_params(self, trial) -> Dict[str, Any]:
        """Suggest continuous parameters"""
        pass
    
    @abstractmethod
    def suggest_model_specific_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest model-specific parameters"""
        pass
    
    def suggest_all_params(self, trial) -> Dict[str, Any]:
        """Combine all parameter suggestions"""
        params = {}
        
        # Categorical parameters
        params.update(self.suggest_categorical_params(trial))

        # Boolean parameters
        params.update(self.suggest_boolean_params(trial))
        
        # Continuous parameters
        params.update(self.suggest_continuous_params(trial))

        params.update(self.suggest_model_specific_params(trial, params.get('model_type')))

        return params
    
    def evaluate_model(self, 
                      model: ScOPE,
                      X_samples: List[str],
                      y_true: List[str],
                      kw_samples_list: List[Dict[str, Any]],
                      cv_folds: int = 3) -> Dict[str, float]:
        """Evaluate the model using cross-validation."""
        
        indices = np.arange(len(X_samples))
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'f1_score': [],
            'auc_roc': [],
            'log_loss': []
        }
        
        unique_classes = sorted(list(set(y_true)))
        if len(unique_classes) != 2:
            raise ValueError(f"Expected exactly 2 classes, but found {len(unique_classes)}: {unique_classes}")
        
        class_to_idx = {unique_classes[0]: 0, unique_classes[1]: 1}
        
        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_true)):
                X_val = [X_samples[i] for i in val_idx]
                y_val = [y_true[i] for i in val_idx]
                kw_val = [kw_samples_list[i] for i in val_idx]
                
                y_pred = []
                y_pred_proba = []
                
                for sample, kw_sample in zip(X_val, kw_val):
                    try:
                        
                        predictions = model.__forward__(sample, kw_sample)

                        softmax_probs = predictions.get('softmax', {})

                        class_names = sorted(softmax_probs.keys())
                        proba_values = [softmax_probs[cls] for cls in class_names]
                        predicted_class_idx = np.argmax(proba_values)
                        
                        y_pred.append(predicted_class_idx)
                        y_pred_proba.append(proba_values)
                        
                    except Exception as e:
                        print(f"Error en predicción individual: {e}")
                        # Handle error by using random predictions
                        random_pred = np.random.randint(0, 2)
                        y_pred.append(random_pred)
                        
                        # Generate random probabilities
                        random_proba = np.random.dirichlet([1, 1])
                        y_pred_proba.append(random_proba.tolist())

                y_val_numeric = np.array([class_to_idx[cls] for cls in y_val])
                y_pred_numeric = np.array(y_pred)
                y_pred_proba_array = np.array(y_pred_proba)

                if len(set(y_pred_numeric)) > 1 and len(set(y_val_numeric)) > 1:
                    try:
                        report = make_report(y_val_numeric, y_pred_numeric, y_pred_proba_array)
                        
                        cv_scores['accuracy'].append(report['acc'])
                        cv_scores['f1_score'].append(report['f1_score'])
                        cv_scores['auc_roc'].append(report['auc_roc'])
                        cv_scores['log_loss'].append(report['log_loss'])
                        
                    except Exception as e:
                        print(f"Error en make_report: {e}")
                        cv_scores['accuracy'].append(0.0)
                        cv_scores['f1_score'].append(0.0)
                        cv_scores['auc_roc'].append(0.5)
                        cv_scores['log_loss'].append(1.0)
                else:
                    cv_scores['accuracy'].append(0.0)
                    cv_scores['f1_score'].append(0.0)
                    cv_scores['auc_roc'].append(0.5)
                    cv_scores['log_loss'].append(1.0)
        
        except Exception as e:
            print(f"Error en evaluación: {e}")
            return {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.5,
                'log_loss': 1.0
            }
        
        return {
            metric: np.mean(scores) for metric, scores in cv_scores.items()
        }
    
    @staticmethod
    def optimize(self, *args, **kwargs) -> Any:
        raise NotImplementedError("This method should be implemented in subclasses.")