import os
import pickle
import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from sklearn.model_selection import StratifiedKFold

from scope.model import ScOPE
from scope.utils.report_generation import make_report

from scope.utils.optimize.params import ParameterSpace, ObjectiveConfig, OptimizationDirection

warnings.filterwarnings('ignore')



class ScOPEOptimizer(ABC):
    """Improved abstract base class for ScOPE model optimizers with better Optuna integration."""

    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None, 
                 objectives: Optional[List[Union[str, ObjectiveConfig]]] = None,
                 free_cpu: int = 0, 
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 evaluation_timeout: Optional[float] = None,
                 fail_fast: bool = True):
        """
        Initialize the optimizer.
        
        Args:
            parameter_space: Parameter space configuration
            objectives: List of objectives (strings or ObjectiveConfig instances)
            free_cpu: Number of CPUs to keep free
            random_seed: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
            evaluation_timeout: Timeout for single model evaluation
            fail_fast: Whether to fail fast on evaluation errors
        """
        self.parameter_space = parameter_space or ParameterSpace()
        self.parameter_space.validate()
        
        self.n_jobs = max(1, os.cpu_count() - free_cpu)
        self.random_seed = random_seed
        self.cv_folds = cv_folds
        self.evaluation_timeout = evaluation_timeout
        self.fail_fast = fail_fast
        
        # Set up objectives
        self.objectives = self._setup_objectives(objectives)
        
        # Cache for expensive computations
        self._evaluation_cache = {}
        
        # Statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'cache_hits': 0
        }
    
    def _setup_objectives(self, objectives: Optional[List[Union[str, ObjectiveConfig]]]) -> List[ObjectiveConfig]:
        """Setup and validate objectives configuration."""
        if objectives is None:
            objectives = ['auc_roc']  # Default single objective
        
        objective_configs = []
        for obj in objectives:
            if isinstance(obj, str):
                # Convert string to ObjectiveConfig with default settings
                direction = OptimizationDirection.MINIMIZE if obj == 'log_loss' else OptimizationDirection.MAXIMIZE
                objective_configs.append(ObjectiveConfig(name=obj, direction=direction))
            elif isinstance(obj, ObjectiveConfig):
                objective_configs.append(obj)
            else:
                raise ValueError(f"Invalid objective type: {type(obj)}. Must be str or ObjectiveConfig")
        
        # Validate objectives
        valid_metrics = {'f1_score', 'auc_roc', 'log_loss', 'accuracy', 'precision', 'recall'}
        for obj_config in objective_configs:
            if obj_config.name not in valid_metrics:
                raise ValueError(f"Invalid objective '{obj_config.name}'. Valid options: {valid_metrics}")
        
        return objective_configs
    
    def create_model_from_params(self, params: Dict[str, Any]) -> ScOPE:
        """Create a ScOPE model instance from the given parameters."""
        # Common base parameters
        base_params = {
            'compressor_name': params['compressor_name'],
            'compression_metric': params['compression_metric'],
            'compression_level': params['compression_level'],
            'min_size_threshold': params['min_size_threshold'],
            'use_best_sigma': params['use_best_sigma'],
            'symetric_matrix': params['symetric_matrix'],
            'string_separator': params['string_separator'],
            'model_type': params['model_type'],
            'use_softmax': True
        }

        # Model-specific parameters based on type
        model_kwargs = {}
        
        if params['model_type'] == "ot":
            model_kwargs['use_matching_method'] = params.get('ot_use_matching_method', False)
            
            if model_kwargs['use_matching_method']:
                model_kwargs['matching_method_name'] = params.get('ot_matching_method_name', 'matching')
                
        elif params['model_type'] == "pd":
            model_kwargs['distance_metric'] = params.get('pd_distance_metric', 'cosine')
            model_kwargs['use_prototypes'] = params.get('pd_use_prototypes', False)

        all_params = {**base_params, **model_kwargs}
        
        return ScOPE(**all_params)
    
    def suggest_categorical_params(self, trial) -> Dict[str, Any]:
        """Suggest categorical parameters with improved validation."""
        params = {
            'compressor_name': trial.suggest_categorical(
                'compressor_name',
                self.parameter_space.compressor_names
            ),
            'compression_metric': trial.suggest_categorical(
                'compression_metric',
                self.parameter_space.compression_metrics
            ),
            'string_separator': trial.suggest_categorical(
                'string_separator', 
                self.parameter_space.string_separators
            ),
            'model_type': trial.suggest_categorical(
                'model_type',
                self.parameter_space.model_types
            )
        }
        
        return params
    
    def suggest_boolean_params(self, trial) -> Dict[str, Any]:
        """Suggest boolean parameters."""
        return {
            'use_best_sigma': trial.suggest_categorical(
                'use_best_sigma', 
                self.parameter_space.use_best_sigma_options
            ),
            'symetric_matrix': trial.suggest_categorical(
                'symetric_matrix',
                self.parameter_space.symetric_matrix_options
            )
        }
    
    def suggest_continuous_params(self, trial) -> Dict[str, Any]:
        """Suggest continuous/discrete parameters."""
        return {
            'compression_level': trial.suggest_categorical(
                'compression_level',
                self.parameter_space.compression_levels
            ),
            'min_size_threshold': trial.suggest_categorical(
                'min_size_threshold',
                self.parameter_space.min_size_thresholds
            )
        }
    
    def suggest_model_specific_params(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest model-specific parameters with conditional logic."""        
        params = {}
                
        if model_type == "ot":
            # OT parameters
            if self.parameter_space.ot_use_matching_method_options:
                params['ot_use_matching_method'] = trial.suggest_categorical(
                    'ot_use_matching_method',
                    self.parameter_space.ot_use_matching_method_options
                )
                
                # Only suggest matching_method_name if use_matching_method might be True
                if (True in self.parameter_space.ot_use_matching_method_options and 
                    self.parameter_space.ot_matching_method_names):
                    
                    # Use conditional parameter suggestion
                    if params.get('ot_use_matching_method', False):
                        params['ot_matching_method_name'] = trial.suggest_categorical(
                            'ot_matching_method_name',
                            self.parameter_space.ot_matching_method_names
                        )
            
        elif model_type == "pd":
            # PD parameters
            if self.parameter_space.pd_distance_metrics:
                params['pd_distance_metric'] = trial.suggest_categorical(
                    'pd_distance_metric',
                    self.parameter_space.pd_distance_metrics
                )
            
            if self.parameter_space.pd_use_prototypes_options:
                params['pd_use_prototypes'] = trial.suggest_categorical(
                    'pd_use_prototypes',
                    self.parameter_space.pd_use_prototypes_options
                )
        
        return params
    
    def suggest_all_params(self, trial) -> Dict[str, Any]:
        """Combine all parameter suggestions with validation."""
        params = {}
        
        # Get categorical parameters first (includes model_type)
        params.update(self.suggest_categorical_params(trial))
        
        # Boolean parameters
        params.update(self.suggest_boolean_params(trial))
        
        # Continuous parameters
        params.update(self.suggest_continuous_params(trial))

        # Model-specific parameters based on selected model type
        model_type = params.get('model_type')
        if model_type:
            params.update(self.suggest_model_specific_params(trial, model_type))

        return params
    
    def _get_params_hash(self, params: Dict[str, Any]) -> str:
        """Generate a hash for parameter combination for caching."""
        import hashlib
        # Create a sorted string representation of parameters
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def evaluate_model(self, 
                      model: ScOPE,
                      X_samples: List[str],
                      y_true: List[str],
                      kw_samples_list: List[Dict[str, Any]],
                      use_cache: bool = True
                      ) -> Dict[str, float]:
        """
        Evaluate the model using cross-validation with improved error handling and caching.
        """
        self.evaluation_stats['total_evaluations'] += 1
        
        # Generate cache key if caching is enabled
        cache_key = None
        if use_cache:
            model_params = {
                # Basic parameters - usar los nombres correctos con _
                'compressor_name': getattr(model, '_compressor_name', None),
                'compression_metric': getattr(model, '_compression_metric', None),
                'compression_level': getattr(model, '_compression_level', None),
                'min_size_threshold': getattr(model, '_min_size_threshold', None),
                'string_separator': getattr(model, '_string_separator', None),
                'use_best_sigma': getattr(model, '_using_sigma', None),
                'symetric_matrix': getattr(model, '_symetric_matrix', None),
                'model_type': getattr(model, '_model_type', None),
                
                # Model kwargs (incluye parámetros específicos de OT/PD)
                'model_kwargs': getattr(model, '_model_kwargs', {}),
            }
            
            cache_key = self._get_params_hash(model_params)
            
            if cache_key in self._evaluation_cache:
                self.evaluation_stats['cache_hits'] += 1
                return self._evaluation_cache[cache_key]
        
        indices = np.arange(len(X_samples))
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        cv_scores = {
            'f1_score': [],
            'auc_roc': [],
            'log_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        
        unique_classes = sorted(list(set(y_true)))
        if len(unique_classes) != 2:
            error_msg = f"Expected exactly 2 classes, but found {len(unique_classes)}: {unique_classes}"
            if self.fail_fast:
                raise ValueError(error_msg)
            else:
                warnings.warn(error_msg)
                return self._get_default_scores()
        
        class_to_idx = {unique_classes[0]: 0, unique_classes[1]: 1}
        
        try:
            for fold, (train_idx, val_idx) in enumerate(skf.split(indices, y_true)):
                X_val = [X_samples[i] for i in val_idx]
                y_val = [y_true[i] for i in val_idx]
                kw_val = [kw_samples_list[i] for i in val_idx]
                
                fold_scores = self._evaluate_fold(model, X_val, y_val, kw_val, class_to_idx, fold)
                
                # Accumulate fold scores
                for metric, score in fold_scores.items():
                    if metric in cv_scores:
                        cv_scores[metric].append(score)
        
        except Exception as e:
            error_msg = f"Error in cross-validation evaluation: {e}"
            if self.fail_fast:
                raise RuntimeError(error_msg) from e
            else:
                warnings.warn(error_msg)
                self.evaluation_stats['failed_evaluations'] += 1
                return self._get_default_scores()
        
        # Calculate mean scores
        final_scores = {}
        for metric, scores in cv_scores.items():
            if scores:  # Only include metrics that have scores
                final_scores[metric] = np.mean(scores)
            else:
                final_scores[metric] = self._get_default_score_for_metric(metric)
        
        # Cache results
        if use_cache and cache_key:
            self._evaluation_cache[cache_key] = final_scores
        
        self.evaluation_stats['successful_evaluations'] += 1
        return final_scores
    
    def _evaluate_fold(self, model: ScOPE, X_val: List[str], y_val: List[str], 
                      kw_val: List[Dict], class_to_idx: Dict[str, int], fold: int) -> Dict[str, float]:
        """Evaluate a single fold with improved error handling."""
        y_pred = []
        y_pred_proba = []
        
        for i, (sample, kw_sample) in enumerate(zip(X_val, kw_val)):
            try:
                predictions = model.__forward__(sample, kw_sample)
                
                if isinstance(predictions, dict) and 'softmax' in predictions:
                    softmax_probs = predictions['softmax']
                    class_names = sorted(softmax_probs.keys())
                    proba_values = [softmax_probs[cls] for cls in class_names]
                    
                    predicted_class_idx = np.argmax(proba_values)
                    y_pred.append(predicted_class_idx)
                    y_pred_proba.append(proba_values)
                else:
                    # Fallback for non-standard prediction format
                    predicted_class_idx = 0 if predictions < 0.5 else 1
                    y_pred.append(predicted_class_idx)
                    y_pred_proba.append([1-predictions, predictions] if isinstance(predictions, (int, float)) else [0.5, 0.5])
                    
            except Exception as e:
                if self.fail_fast:
                    raise RuntimeError(f"Error in fold {fold}, sample {i}: {e}") from e
                
                # Handle error with random predictions
                random_pred = np.random.randint(0, 2)
                y_pred.append(random_pred)
                random_proba = np.random.dirichlet([1, 1])
                y_pred_proba.append(random_proba.tolist())

        # Convert to numpy arrays
        y_val_numeric = np.array([class_to_idx[cls] for cls in y_val])
        y_pred_numeric = np.array(y_pred)
        y_pred_proba_array = np.array(y_pred_proba)

        # Calculate metrics for this fold
        try:
            if len(set(y_pred_numeric)) > 1 and len(set(y_val_numeric)) > 1:
                report = make_report(y_val_numeric, y_pred_numeric, y_pred_proba_array)
                return {
                    'f1_score': report.get('f1_score', 0.0),
                    'auc_roc': report.get('auc_roc', 0.5),
                    'log_loss': report.get('log_loss', 1.0),
                    'accuracy': report.get('accuracy', 0.0),
                    'precision': report.get('precision', 0.0),
                    'recall': report.get('recall', 0.0)
                }
            else:
                return self._get_default_scores()
                
        except Exception as e:
            if self.fail_fast:
                raise RuntimeError(f"Error calculating metrics for fold {fold}: {e}") from e
            return self._get_default_scores()
    
    def _get_default_scores(self) -> Dict[str, float]:
        """Get default scores for failed evaluations."""
        return {
            'f1_score': 0.0,
            'auc_roc': 0.5,
            'log_loss': 1.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }
    
    def _get_default_score_for_metric(self, metric: str) -> float:
        """Get default score for a specific metric."""
        defaults = self._get_default_scores()
        return defaults.get(metric, 0.0)
    
    def print_parameter_space(self):
        """Print detailed parameter space information."""
        print("Parameter space configuration:")
        print("=" * 60)
        
        # Basic parameters
        print("BASIC PARAMETERS:")
        print(f"  • Compressors ({len(self.parameter_space.compressor_names)}): {self.parameter_space.compressor_names}")
        print(f"  • Compression metrics ({len(self.parameter_space.compression_metrics)}): {self.parameter_space.compression_metrics}")
        print(f"  • Compression levels ({len(self.parameter_space.compression_levels)}): {self.parameter_space.compression_levels}") 
        print(f"  • Min size thresholds ({len(self.parameter_space.min_size_thresholds)}): {self.parameter_space.min_size_thresholds}")
        print(f"  • String separators ({len(self.parameter_space.string_separators)}): {[repr(s) for s in self.parameter_space.string_separators]}")
        print(f"  • Use best sigma: {self.parameter_space.use_best_sigma_options}")
        print(f"  • Use symetric matrix: {self.parameter_space.symetric_matrix_options}")
        print(f"  • Model types: {self.parameter_space.model_types}")
        
        print("\nMODEL-SPECIFIC PARAMETERS:")
        
        # ScOPE-OT parameters
        print("  ScOPE-OT:")
        print(f"    • Use matching method: {self.parameter_space.ot_use_matching_method_options}")
        print(f"    • Matching methods ({len(self.parameter_space.ot_matching_method_names)}): {self.parameter_space.ot_matching_method_names}")
        
        # ScOPE-PD parameters  
        print("  ScOPE-PD:")
        print(f"    • Distance metrics ({len(self.parameter_space.pd_distance_metrics)}): {self.parameter_space.pd_distance_metrics}")
        print(f"    • Use prototypes: {self.parameter_space.pd_use_prototypes_options}")
        
        total_combinations = self.parameter_space.get_total_combinations()
        print(f"\nTOTAL POSSIBLE COMBINATIONS: ~{total_combinations:,}")
        
        # Objectives information
        print("\nOPTIMIZATION OBJECTIVES:")
        for i, obj in enumerate(self.objectives):
            print(f"  {i+1}. {obj.name} ({obj.direction.value}, weight: {obj.weight})")
        
        print("=" * 60)
    
    def print_evaluation_stats(self):
        """Print evaluation statistics."""
        stats = self.evaluation_stats
        total = stats['total_evaluations']
        
        print("\nEvaluation Statistics:")
        print("-" * 30)
        print(f"Total evaluations: {total}")
        print(f"Successful: {stats['successful_evaluations']} ({stats['successful_evaluations']/max(1,total)*100:.1f}%)")
        print(f"Failed: {stats['failed_evaluations']} ({stats['failed_evaluations']/max(1,total)*100:.1f}%)")
        print(f"Cache hits: {stats['cache_hits']} ({stats['cache_hits']/max(1,total)*100:.1f}%)")
        print(f"Cache size: {len(self._evaluation_cache)} entries")
    
    def clear_cache(self):
        """Clear the evaluation cache."""
        self._evaluation_cache.clear()
        print("Evaluation cache cleared.")
    
    def get_objective_value(self, scores: Dict[str, float], objective: ObjectiveConfig) -> float:
        """Get the objective value with proper direction handling."""
        raw_value = scores.get(objective.name, self._get_default_score_for_metric(objective.name))
        
        # Apply weight
        weighted_value = raw_value * objective.weight
        
        # Handle direction (Optuna handles maximize/minimize at study level)
        return weighted_value
    
    def get_objective_values(self, scores: Dict[str, float]) -> List[float]:
        """Get all objective values for multi-objective optimization."""
        return [self.get_objective_value(scores, obj) for obj in self.objectives]
    
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective optimization."""
        return len(self.objectives) > 1
    
    def get_directions(self) -> List[str]:
        """Get optimization directions for Optuna study."""
        return [obj.direction.value for obj in self.objectives]
    
    def get_best_model(self, study: Any = None) -> ScOPE:
        """Get the best optimized model for single-objective optimization."""
        if study is None:
            raise ValueError("Study must be provided")
            
        if self.is_multi_objective():
            raise ValueError("For multi-objective optimization, use get_pareto_front_models() or get_best_compromise_model() instead.")
        
        if not hasattr(study, 'best_params') or study.best_params is None:
            raise ValueError("No optimized model found. Run optimize() first.")
            
        return self.create_model_from_params(study.best_params)
    
    def get_pareto_front_models(self, study: Any) -> List[ScOPE]:
        """Get all Pareto-optimal models for multi-objective optimization."""
        if not self.is_multi_objective():
            raise ValueError("This method is only available for multi-objective optimization.")
        
        if not hasattr(study, 'best_trials') or not study.best_trials:
            raise ValueError("No Pareto front found. Run optimize() first.")
        
        return [self.create_model_from_params(trial.params) for trial in study.best_trials]
    
    def get_best_compromise_model(self, study: Any, weights: Optional[Dict[str, float]] = None) -> ScOPE:
        """Get best compromise model from Pareto front using weighted sum."""
        if not self.is_multi_objective():
            return self.get_best_model(study)
        
        if not hasattr(study, 'best_trials') or not study.best_trials:
            raise ValueError("No Pareto front found. Run optimize() first.")
        
        if weights is None:
            weights = {obj.name: obj.weight for obj in self.objectives}
        
        # Find best compromise solution
        best_trial = None
        best_score = float('-inf')
        
        # Get value ranges for normalization
        all_values = {}
        for obj in self.objectives:
            obj_idx = next(i for i, o in enumerate(self.objectives) if o.name == obj.name)
            all_values[obj.name] = [trial.values[obj_idx] for trial in study.best_trials]
        
        # Calculate min/max for normalization
        obj_ranges = {}
        for obj_name, values in all_values.items():
            obj_ranges[obj_name] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values) if max(values) != min(values) else 1.0
            }
        
        for trial in study.best_trials:
            weighted_score = 0.0
            for i, obj in enumerate(self.objectives):
                value = trial.values[i]
                
                # Normalize to [0, 1]
                if obj.direction == OptimizationDirection.MINIMIZE:
                    # For minimization, invert the normalization
                    normalized = 1.0 - (value - obj_ranges[obj.name]['min']) / obj_ranges[obj.name]['range']
                else:
                    # For maximization
                    normalized = (value - obj_ranges[obj.name]['min']) / obj_ranges[obj.name]['range']
                
                weighted_score += weights.get(obj.name, obj.weight) * normalized
            
            if weighted_score > best_score:
                best_score = weighted_score
                best_trial = trial
        
        if best_trial is None:
            raise ValueError("No suitable compromise solution found.")
        
        return self.create_model_from_params(best_trial.params)
    
    def analyze_results(self, study: Any) -> pd.DataFrame:
        """Analyze optimization results with comprehensive insights."""
        if study is None:
            raise ValueError("Study must be provided")
        
        print("\n=== OPTIMIZATION ANALYSIS ===")
        
        # Import here to avoid circular imports
        import optuna
        
        # Basic statistics
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"Completed trials: {completed_trials}")
        print(f"Pruned trials: {pruned_trials}")
        print(f"Failed trials: {failed_trials}")
        print(f"Success rate: {completed_trials/max(1, len(study.trials))*100:.1f}%")
        
        # Evaluation statistics
        self.print_evaluation_stats()
        
        # Parameter importance (only for single-objective)
        if not self.is_multi_objective():
            try:
                importances = optuna.importance.get_param_importances(study)
                print("\nParameter importance:")
                print("-" * 40)
                
                # Separate basic and model-specific parameters
                basic_params = {}
                ot_params = {}
                pd_params = {}
                
                for param, importance in importances.items():
                    if param.startswith('ot_'):
                        ot_params[param] = importance
                    elif param.startswith('pd_'):
                        pd_params[param] = importance
                    else:
                        basic_params[param] = importance
                
                # Print categorized parameters
                if basic_params:
                    print("Basic Parameters:")
                    for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {param}: {importance:.6f}")
                    print()
                
                if ot_params:
                    print("ScOPE-OT Parameters:")
                    for param, importance in sorted(ot_params.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {param}: {importance:.6f}")
                    print()
                
                if pd_params:
                    print("ScOPE-PD Parameters:")
                    for param, importance in sorted(pd_params.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {param}: {importance:.6f}")
                    print()
                
                print("Overall Ranking (Top 10):")
                top_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                for param, importance in top_params:
                    print(f"  {param}: {importance:.6f}")
                
            except Exception as e:
                print(f"Could not calculate parameter importance: {e}")
        
        # DataFrame with results
        df_results = study.trials_dataframe()
        if not df_results.empty:
            if self.is_multi_objective():
                print(f"\nPareto front analysis:")
                print(f"Total solutions in Pareto front: {len(study.best_trials) if hasattr(study, 'best_trials') else 0}")
            else:
                print("\nTop 5 configurations:")
                columns_to_show = ['value', 'params_compressor_name', 'params_compression_metric', 
                                  'params_use_best_sigma', 'params_symetric_matrix', 'params_model_type']
                
                # Add model-specific columns if they exist
                model_specific_columns = []
                for col in df_results.columns:
                    if col.startswith('params_ot_') or col.startswith('params_pd_'):
                        model_specific_columns.append(col)
                
                columns_to_show.extend(model_specific_columns)
                available_columns = [col for col in columns_to_show if col in df_results.columns]
                
                if available_columns and 'value' in df_results.columns:
                    top_trials = df_results.nlargest(5, 'value')[available_columns]
                    print(top_trials)
        
        return df_results
    
    def save_results(self, study: Any, output_path: str, study_name: str, study_date: str, 
                    filename: Optional[str] = None, **kwargs) -> str:
        """Save optimization results with comprehensive metadata."""
        if study is None:
            raise ValueError("Study must be provided")
        
        os.makedirs(output_path, exist_ok=True)
        
        if filename is None:
            optimization_type = "multi" if self.is_multi_objective() else "single"
            filename = f"{study_name}_{optimization_type}_results_{study_date}.pkl"
        
        filepath = os.path.join(output_path, filename)
        
        results = {
            'study': study,
            'objectives': [{'name': obj.name, 'direction': obj.direction.value, 'weight': obj.weight} 
                          for obj in self.objectives],
            'is_multi_objective': self.is_multi_objective(),
            'parameter_space': self.parameter_space,
            'evaluation_stats': self.evaluation_stats,
            'optimizer_type': self.__class__.__name__
        }
        
        # Add algorithm-specific configuration
        results.update(kwargs)
        
        # Add optimization-type specific results
        if not self.is_multi_objective():
            if hasattr(study, 'best_params') and hasattr(study, 'best_value'):
                results.update({
                    'best_params': study.best_params,
                    'best_value': study.best_value
                })
        else:
            if hasattr(study, 'best_trials'):
                results['pareto_front'] = study.best_trials
        
        results['trials_dataframe'] = study.trials_dataframe()
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def save_top_results_csv(self, study: Any, output_path: str, study_name: str, study_date: str,
                            filename: Optional[str] = None, top_n: int = 10) -> pd.DataFrame:
        """Save top N results to CSV file."""
        if study is None:
            raise ValueError("Study must be provided")
        
        os.makedirs(output_path, exist_ok=True)
        
        if filename is None:
            filename = f"{study_name}_top{top_n}_{study_date}.csv"

        filepath = os.path.join(output_path, filename)
        
        df_results = study.trials_dataframe()
        if df_results.empty:
            print("No results to save.")
            return pd.DataFrame()
        
        # Determine value column based on optimization type
        if self.is_multi_objective():
            # For multi-objective, use first objective as primary sort
            value_col = 'values_0'
        else:
            value_col = 'value'
        
        if value_col not in df_results.columns:
            print(f"Warning: Expected column '{value_col}' not found. Using available value column.")
            value_cols = [col for col in df_results.columns if 'value' in col.lower()]
            if value_cols:
                value_col = value_cols[0]
            else:
                print("No value columns found.")
                return pd.DataFrame()
        
        # Get top N results
        top_results = df_results.nlargest(top_n, value_col)
        
        # Clean column names
        cleaned_df = top_results.copy()
        column_mapping = {}
        for col in cleaned_df.columns:
            if col.startswith('params_'):
                new_name = col.replace('params_', '')
                column_mapping[col] = new_name
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Select relevant columns
        important_cols = ['number', value_col, 'state', 'datetime_start', 'datetime_complete']
        if 'duration' in cleaned_df.columns:
            important_cols.append('duration')
        
        basic_param_cols = ['compressor_name', 'compression_metric', 'compression_level', 
                           'min_size_threshold', 'string_separator', 'use_best_sigma', 'model_type', 'symetric_matrix']
        model_specific_cols = [col for col in cleaned_df.columns if col.startswith(('ot_', 'pd_'))]
        
        all_param_cols = basic_param_cols + model_specific_cols
        final_cols = [col for col in important_cols + all_param_cols if col in cleaned_df.columns]
        
        result_df = cleaned_df[final_cols]
        
        # Add rank column
        result_df.insert(0, 'rank', range(1, len(result_df) + 1))
        
        # Save to CSV
        result_df.to_csv(filepath, index=False)
        print(f"Top {top_n} results saved to {filepath}")
        
        return result_df
    
    def save_pareto_front_csv(self, study: Any, output_path: str, study_name: str, study_date: str,
                             filename: Optional[str] = None) -> pd.DataFrame:
        """Save Pareto front to CSV for multi-objective optimization."""
        if not self.is_multi_objective():
            raise ValueError("This method is only available for multi-objective optimization.")
        
        if not hasattr(study, 'best_trials') or not study.best_trials:
            print("No Pareto front solutions to save.")
            return pd.DataFrame()
        
        os.makedirs(output_path, exist_ok=True)
        
        if filename is None:
            filename = f"{study_name}_pareto_front_{study_date}.csv"
        
        filepath = os.path.join(output_path, filename)
        
        # Create DataFrame with Pareto front solutions
        pareto_data = []
        for i, trial in enumerate(study.best_trials):
            row = {'rank': i + 1, 'trial_number': trial.number}
            
            # Add objective values
            for j, obj in enumerate(self.objectives):
                row[obj.name] = trial.values[j]
            
            # Add parameters
            for param, value in trial.params.items():
                if param == 'string_separator':
                    value = repr(value)
                row[param] = value
            
            pareto_data.append(row)
        
        pareto_df = pd.DataFrame(pareto_data)
        pareto_df.to_csv(filepath, index=False)
        
        print(f"Pareto front saved to {filepath}")
        return pareto_df
    
    def print_optimization_results(self, study: Any):
        """Print optimization results with enhanced formatting."""
        if study is None:
            raise ValueError("Study must be provided")
        
        print("\n=== OPTIMIZATION RESULTS ===")
        
        if self.is_multi_objective():
            if hasattr(study, 'best_trials'):
                pareto_solutions = len(study.best_trials)
                print(f"Pareto-optimal solutions found: {pareto_solutions}")
                
                if pareto_solutions > 0:
                    print("\nTop 3 Pareto-optimal solutions:")
                    for i, trial in enumerate(study.best_trials[:3]):
                        print(f"\nSolution {i+1} (Trial #{trial.number}):")
                        for j, obj in enumerate(self.objectives):
                            print(f"  {obj.name}: {trial.values[j]:.4f}")
                        
                        # Show key parameters
                        key_params = ['compressor_name', 'compression_metric', 'model_type']
                        print("  Key parameters:")
                        for param in key_params:
                            if param in trial.params:
                                value = trial.params[param]
                                if param == 'string_separator':
                                    value = repr(value)
                                print(f"    {param}: {value}")
            else:
                print("No Pareto front solutions found.")
        else:
            if hasattr(study, 'best_value') and hasattr(study, 'best_params'):
                print(f"Best score: {study.best_value:.4f}")
                print("Best parameters:")
                for param, value in study.best_params.items():
                    if param == 'string_separator':
                        value = repr(value)
                    print(f"  {param}: {value}")
            else:
                print("No best solution found.")

    @abstractmethod
    def optimize(self, X_validation: List[str], y_validation: List[str], 
                kw_samples_validation: List[Dict[str, List[str]]]) -> Any:
        """Run optimization (to be implemented by subclasses)."""
        pass


# Example usage and testing
if __name__ == "__main__":
    # Test the improved base class
    
    # Custom parameter space
    custom_space = ParameterSpace(
        compressor_names=['gzip', 'bz2'],
        compression_metrics=['ncd', 'ncs'],
        model_types=['ot', 'pd'],
        compression_levels=[1, 5, 9]
    )
    
    # Custom objectives
    objectives = [
        ObjectiveConfig('auc_roc', OptimizationDirection.MAXIMIZE, weight=0.6),
        ObjectiveConfig('f1_score', OptimizationDirection.MAXIMIZE, weight=0.4)
    ]
    
    # This would be implemented by a concrete subclass
    class TestOptimizer(ScOPEOptimizer):
        def optimize(self, X_validation, y_validation, kw_samples_validation):
            print("This would run the actual optimization...")
            return None
    
    optimizer = TestOptimizer(
        parameter_space=custom_space,
        objectives=objectives,
        cv_folds=3,
        fail_fast=False
    )
    
    optimizer.print_parameter_space()
    print(f"\nMulti-objective: {optimizer.is_multi_objective()}")
    print(f"Directions: {optimizer.get_directions()}")