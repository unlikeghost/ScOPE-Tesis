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
    """
    Improved Bayesian optimization for ScOPE models using Optuna.
    Supports flexible objective configuration: string, dict, or list.
    """
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 objectives: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any], ObjectiveConfig]]] = 'auc_roc',
                 free_cpu: int = 0,
                 n_trials: int = 50,
                 timeout: int = 1800,
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_bayesian_optimization",
                 output_path: str = "./results",
                 sampler_config: Optional[Dict[str, Any]] = None,
                 pruner_config: Optional[Dict[str, Any]] = None,
                 use_cache: bool = True,
                 fail_fast: bool = False):
        """
        Initialize the Bayesian optimizer with flexible objective configuration.
        
        Args:
            objectives: Can be:
                - String: 'auc_roc', 'f1_score', etc.
                - Dict: {'auc_roc': {'direction': 'maximize', 'weight': 0.7}}
                - List: Mix of strings, dicts, or ObjectiveConfig
        """
        
        # Convert objectives to ObjectiveConfig format
        processed_objectives = self._process_objectives(objectives)
        
        super().__init__(
            parameter_space=parameter_space,
            objectives=processed_objectives,
            free_cpu=free_cpu,
            random_seed=random_seed,
            cv_folds=cv_folds,
            fail_fast=fail_fast
        )
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name
        self.output_path = output_path
        self.use_cache = use_cache
        
        # Sampler and pruner configurations
        self.sampler_config = sampler_config or {
            'n_startup_trials': 10,
            'n_ei_candidates': 24
        }
        self.pruner_config = pruner_config or {
            'n_startup_trials': 5,
            'n_warmup_steps': 1
        }
        
        os.makedirs(self.output_path, exist_ok=True)
        
        self.study_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store results
        self.study = None
        self.best_params = None
        self.best_model = None
    
    def _process_objectives(self, objectives) -> List[ObjectiveConfig]:
        """
        Process different objective formats into ObjectiveConfig list.
        
        Supports:
        - String: 'auc_roc'
        - Dict: {'auc_roc': {'direction': 'maximize', 'weight': 0.7}}
        - List: ['auc_roc', {'f1_score': {'direction': 'maximize'}}]
        """
        if isinstance(objectives, str):
            # Single string objective
            direction = OptimizationDirection.MINIMIZE if objectives == 'log_loss' else OptimizationDirection.MAXIMIZE
            return [ObjectiveConfig(name=objectives, direction=direction, weight=1.0)]
        
        elif isinstance(objectives, dict):
            # Dictionary format: {'metric': {'direction': 'maximize', 'weight': 0.7}}
            obj_configs = []
            for metric_name, config in objectives.items():
                if isinstance(config, dict):
                    direction_str = config.get('direction', 'maximize' if metric_name != 'log_loss' else 'minimize')
                    direction = OptimizationDirection.MAXIMIZE if direction_str.lower() == 'maximize' else OptimizationDirection.MINIMIZE
                    weight = config.get('weight', 1.0)
                else:
                    # Simple format: {'auc_roc': 0.7} (weight only)
                    direction = OptimizationDirection.MINIMIZE if metric_name == 'log_loss' else OptimizationDirection.MAXIMIZE
                    weight = float(config)
                
                obj_configs.append(ObjectiveConfig(name=metric_name, direction=direction, weight=weight))
            return obj_configs
        
        elif isinstance(objectives, list):
            # List format: mix of strings, dicts, and ObjectiveConfig
            obj_configs = []
            for obj in objectives:
                if isinstance(obj, str):
                    direction = OptimizationDirection.MINIMIZE if obj == 'log_loss' else OptimizationDirection.MAXIMIZE
                    obj_configs.append(ObjectiveConfig(name=obj, direction=direction, weight=1.0))
                elif isinstance(obj, dict):
                    # Process dict in list
                    for metric_name, config in obj.items():
                        if isinstance(config, dict):
                            direction_str = config.get('direction', 'maximize' if metric_name != 'log_loss' else 'minimize')
                            direction = OptimizationDirection.MAXIMIZE if direction_str.lower() == 'maximize' else OptimizationDirection.MINIMIZE
                            weight = config.get('weight', 1.0)
                        else:
                            direction = OptimizationDirection.MINIMIZE if metric_name == 'log_loss' else OptimizationDirection.MAXIMIZE
                            weight = float(config)
                        obj_configs.append(ObjectiveConfig(name=metric_name, direction=direction, weight=weight))
                elif isinstance(obj, ObjectiveConfig):
                    obj_configs.append(obj)
                else:
                    raise ValueError(f"Invalid objective type in list: {type(obj)}")
            return obj_configs
        
        else:
            raise ValueError(f"Invalid objectives type: {type(objectives)}")


# Example usage with different objective formats
if __name__ == "__main__":
    import numpy as np
    
    # Example data
    np.random.seed(42)
    X_validation = [f"example text {i}" for i in range(100)]
    y_validation = np.random.choice(['class_0', 'class_1'], size=100).tolist()
    kw_samples_validation = []
    for _ in range(100):
        kw_sample = {
            'class_0': [f"example 0 {i}" for i in range(3)],
            'class_1': [f"example 1 {i}" for i in range(3)]
        }
        kw_samples_validation.append(kw_sample)
    
    # Example 1: Simple string
    print("=== SINGLE STRING OBJECTIVE ===")
    optimizer1 = ScOPEOptimizerBayesian(objectives='auc_roc')
    
    # Example 2: Dictionary with direction and weight
    print("=== DICTIONARY OBJECTIVES ===")
    dict_objectives = {
        'auc_roc': {'direction': 'maximize', 'weight': 0.7},
        'f1_score': {'direction': 'maximize', 'weight': 0.3}
    }
    optimizer2 = ScOPEOptimizerBayesian(objectives=dict_objectives)
    
    # Example 3: Dictionary with just weights (direction auto-detected)
    print("=== DICTIONARY WITH WEIGHTS ONLY ===")
    weight_objectives = {
        'auc_roc': 0.6,
        'f1_score': 0.3,
        'log_loss': 0.1  # auto-detected as minimize
    }
    optimizer3 = ScOPEOptimizerBayesian(objectives=weight_objectives)
    
    # Example 4: Mixed list format
    print("=== MIXED LIST OBJECTIVES ===")
    mixed_objectives = [
        'auc_roc',  # String
        {'f1_score': {'direction': 'maximize', 'weight': 0.3}},  # Dict
        ObjectiveConfig('log_loss', OptimizationDirection.MINIMIZE, weight=0.1)  # ObjectiveConfig
    ]
    optimizer4 = ScOPEOptimizerBayesian(objectives=mixed_objectives)
    
    print("All optimizers created successfully!")
    for i, opt in enumerate([optimizer1, optimizer2, optimizer3, optimizer4], 1):
        print(f"Optimizer {i} objectives:")
        for obj in opt.objectives:
            print(f"  - {obj.name}: {obj.direction.value}, weight={obj.weight}")
        print()import os
import pickle
import warnings
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from scope.model import ScOPE
from scope.utils.optimize.base import ScOPEOptimizer, ParameterSpace

warnings.filterwarnings('ignore')


class ScOPEOptimizerBayesian(ScOPEOptimizer):
    """Bayesian optimization for ScOPE models using Optuna with configurable combined metric."""
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 free_cpu: int = 0,
                 n_trials: int = 50,
                 timeout: int = 1800,
                 target_metric: Union[str, Dict[str, Any]] = 'combined',
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_optimization",
                 output_path: str = "./results",
                 # Configurable weights as dictionary (only for 'combined')
                 metric_weights: Optional[Dict[str, float]] = None,
                 use_cache: bool = True):
        """Initialize the Bayesian optimizer with configurable weights"""
        super().__init__(parameter_space, free_cpu, random_seed=random_seed)
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.target_metric = target_metric
        self.study_name = study_name
        self.output_path = output_path
        self.cv_folds = cv_folds
        self.use_cache = use_cache
        
        # Default weights for combined metric
        if metric_weights is None:
            self.metric_weights = {
                'auc_roc': 0.50,
                'f1_score': 0.40,
                'log_loss': 0.10
            }
        else:
            self.metric_weights = metric_weights.copy()
        
        # Validate weights for combined metric
        if target_metric == 'combined':
            total_weight = sum(self.metric_weights.values())
            if abs(total_weight - 1.0) > 0.001:
                warnings.warn(f"Weights sum to {total_weight:.3f}, not 1.0. Consider normalizing.")
            
            # Ensure required metrics are present
            required_metrics = ['auc_roc', 'f1_score', 'log_loss']
            missing_metrics = [m for m in required_metrics if m not in self.metric_weights]
            if missing_metrics:
                raise ValueError(f"Missing required metrics for combined target: {missing_metrics}")
        
        os.makedirs(self.output_path, exist_ok=True)
        
        self.study_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store results
        self.study = None
        self.best_params = None
        self.best_model = None
    
    def _create_objective_function(self,
                                  X_validation: List[str],
                                  y_validation: List[str], 
                                  kw_samples_validation: List[Dict[str, List[str]]]):
        """Create objective function for Optuna"""
        
        def objective(trial):
            try:
                # Use the suggest_all_params method from parent class
                params = self.suggest_all_params(trial)
                
                # Create model with suggested configuration
                model = self.create_model_from_params(params)
                
                # Evaluate configuration using cross-validation
                scores = self.evaluate_model(
                    model=model,
                    X_samples=X_validation,
                    y_true=y_validation,
                    kw_samples_list=kw_samples_validation,
                    cv_folds=self.cv_folds
                )
                
                # Calculate target metric
                if self.target_metric == 'combined':
                    final_score = 0.0
                    for metric, weight in self.metric_weights.items():
                        if metric == 'log_loss':
                            # Normalize log_loss for combination (invert it)
                            normalized_value = 1 / (1 + scores[metric])
                        else:
                            normalized_value = scores[metric]
                        final_score += normalized_value * weight
                elif self.target_metric == 'log_loss':
                    final_score = -scores['log_loss']  # Minimize log_loss
                else:
                    final_score = scores[self.target_metric]
                
                return final_score
                
            except Exception as e:
                print(f"Error in trial: {e}")
                import traceback
                traceback.print_exc()
                return 0.0 if self.target_metric != 'log_loss' else 10.0

        return objective

    def optimize(self,
                X_validation: List[str],
                y_validation: List[str],
                kw_samples_validation: List[Dict[str, List[str]]]) -> optuna.Study:
        """Run Bayesian optimization"""
        
        print("=== BAYESIAN OPTIMIZATION SCOPE ===\n")
        print(f"Validation data: {len(X_validation)} samples")
        print(f"Classes: {sorted(set(y_validation))}")
        print(f"Target metric: {self.target_metric}")
        if self.target_metric == 'combined':
            print("Combined metric weights:")
            for metric, weight in self.metric_weights.items():
                print(f"  {metric}: {weight:.3f}")
        print(f"Trials: {self.n_trials}, Timeout: {self.timeout}s")
        print(f"CV Folds: {self.cv_folds}")
        print(f"Cache enabled: {self.use_cache}\n")
        
        # Validate binary classification
        unique_classes = set(y_validation)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
        
        # Print detailed parameter space
        self.print_parameter_space()
        
        # Create objective function
        objective_func = self._create_objective_function(
            X_validation, y_validation, kw_samples_validation
        )
        
        # Configure Optuna study
        direction = 'maximize' if self.target_metric != 'log_loss' else 'minimize'
        self.study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(
                seed=self.random_seed,
                n_startup_trials=10,
                n_ei_candidates=24
            ),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1
            ),
            study_name=self.study_name
        )
        
        # Execute optimization
        print("\nStarting optimization...")
        self.study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )
        
        # Store best results
        self.best_params = self.study.best_params
        self.best_model = self.create_model_from_params(self.best_params)
        
        # Show results
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Best score: {self.study.best_value:.4f}")
        if self.target_metric == 'combined':
            print("Metric weights used:")
            for metric, weight in self.metric_weights.items():
                print(f"  {metric}: {weight:.3f}")
        print("Best parameters:")
        for param, value in self.best_params.items():
            if param == 'string_separator':
                value = repr(value)
            print(f"  {param}: {value}")
        
        return self.study
    
    def get_best_model(self) -> ScOPE:
        """Get the best optimized model"""
        if self.best_model is None:
            raise ValueError("No optimized model found. Run optimize() first.")
        return self.best_model

    def save_results(self, filename: Optional[str] = None):
        """Save optimization results"""
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_results_{self.study_date}.pkl"
        
        filepath = os.path.join(self.output_path, filename)
        
        results = {
            'study': self.study,
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'target_metric': self.target_metric,
            'metric_weights': self.metric_weights,
            'trials_dataframe': self.study.trials_dataframe()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")

    def analyze_results(self):
        """Analyze optimization results"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        print("\n=== DETAILED ANALYSIS ===")
        
        # Basic statistics
        completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print(f"Completed trials: {completed_trials}")
        print(f"Pruned trials: {pruned_trials}")
        print(f"Failed trials: {failed_trials}")
        
        # Show metric weights
        if self.target_metric == 'combined':
            print(f"\nCombined metric weights:")
            for metric, weight in self.metric_weights.items():
                print(f"  {metric}: {weight:.3f}")
        
        # Parameter importance
        try:
            importances = optuna.importance.get_param_importances(self.study)
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
            
            # Print basic parameters
            if basic_params:
                print("Basic Parameters:")
                for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print ScOPE-OT parameters
            if ot_params:
                print("ScOPE-OT Parameters:")
                for param, importance in sorted(ot_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print ScOPE-PD parameters
            if pd_params:
                print("ScOPE-PD Parameters:")
                for param, importance in sorted(pd_params.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param}: {importance:.6f}")
                print()
            
            # Print overall ranking
            print("Overall Ranking (Top 10):")
            top_params = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for param, importance in top_params:
                print(f"  {param}: {importance:.6f}")
            
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
        
        # DataFrame with results
        df_results = self.study.trials_dataframe()
        if not df_results.empty:
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
            top_trials = df_results.nlargest(5, 'value')[available_columns]
            print(top_trials)
        
        return df_results
    
    def save_analysis_report(self, filename: Optional[str] = None):
        """Save detailed analysis report to text file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_analysis_{self.study_date}.txt"

        filepath = os.path.join(self.output_path, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCOPE BAYESIAN OPTIMIZATION ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Study information
            f.write("STUDY INFORMATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Study name: {self.study_name}\n")
            f.write(f"Study date: {self.study_date}\n")
            f.write(f"Output path: {self.output_path}\n\n")
            
            # Study configuration
            f.write("OPTIMIZATION CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Target metric: {self.target_metric}\n")
            if self.target_metric == 'combined':
                f.write("Metric weights:\n")
                for metric, weight in self.metric_weights.items():
                    f.write(f"  {metric}: {weight:.3f}\n")
            f.write(f"Number of trials: {self.n_trials}\n")
            f.write(f"Timeout: {self.timeout} seconds\n")
            f.write(f"CV folds: {self.cv_folds}\n")
            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Cache enabled: {self.use_cache}\n")
            f.write(f"Best score achieved: {self.study.best_value:.6f}\n\n")
            
            # Parameter space
            f.write("PARAMETER SPACE:\n")
            f.write("-" * 30 + "\n")
            f.write("BASIC PARAMETERS:\n")
            f.write(f"  Compressors ({len(self.parameter_space.compressor_names)}): {self.parameter_space.compressor_names}\n")
            f.write(f"  Compression metrics ({len(self.parameter_space.compression_metrics)}): {self.parameter_space.compression_metrics}\n")
            f.write(f"  Compression levels ({len(self.parameter_space.compression_levels)}): {self.parameter_space.compression_levels}\n")
            f.write(f"  Min size thresholds ({len(self.parameter_space.min_size_thresholds)}): {self.parameter_space.min_size_thresholds}\n")
            f.write(f"  String separators ({len(self.parameter_space.string_separators)}): {self.parameter_space.string_separators}\n")
            f.write(f"  Use best sigma options: {self.parameter_space.use_best_sigma_options}\n")
            f.write(f"  Use symetric matrix options: {self.parameter_space.symetric_matrix_options}\n")
            f.write(f"  Model types: {self.parameter_space.model_types}\n\n")
            
            f.write("MODEL-SPECIFIC PARAMETERS:\n")
            f.write(f"  ScOPE-OT:\n")
            f.write(f"    Use matching method: {self.parameter_space.ot_use_matching_method_options}\n")
            f.write(f"    Matching methods ({len(self.parameter_space.ot_matching_method_names)}): {self.parameter_space.ot_matching_method_names}\n")
            f.write(f"  ScOPE-PD:\n")
            f.write(f"    Distance metrics ({len(self.parameter_space.pd_distance_metrics)}): {self.parameter_space.pd_distance_metrics}\n")
            f.write(f"    Use prototypes: {self.parameter_space.pd_use_prototypes_options}\n\n")
            
            # Trial statistics
            completed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
            failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
            
            f.write("TRIAL STATISTICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Completed trials: {completed_trials}\n")
            f.write(f"Pruned trials: {pruned_trials}\n")
            f.write(f"Failed trials: {failed_trials}\n")
            f.write(f"Total trials: {len(self.study.trials)}\n\n")
            
            # Best parameters
            f.write("BEST CONFIGURATION:\n")
            f.write("-" * 30 + "\n")
            for param, value in self.best_params.items():
                if param == 'string_separator':
                    value = repr(value)
                f.write(f"{param}: {value}\n")
            f.write("\n")
        
        print(f"Analysis report saved to {filepath}")
    
    def save_top_results_csv(self, filename: Optional[str] = None, top_n: int = 10):
        """Save top N results to CSV file"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        if filename is None:
            filename = f"{self.study_name}_top{top_n}_{self.study_date}.csv"

        filepath = os.path.join(self.output_path, filename)
        
        df_results = self.study.trials_dataframe()
        if df_results.empty:
            print("No results to save.")
            return
        
        # Get top N results
        top_results = df_results.nlargest(top_n, 'value')
        
        # Clean up column names (remove 'params_' prefix)
        cleaned_df = top_results.copy()
        column_mapping = {}
        for col in cleaned_df.columns:
            if col.startswith('params_'):
                new_name = col.replace('params_', '')
                column_mapping[col] = new_name
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Select relevant columns
        important_cols = ['number', 'value', 'state']
        basic_param_cols = ['compressor_name', 'compression_metric', 'compression_level', 
                           'min_size_threshold', 'string_separator', 'use_best_sigma', 'model_type', 'symetric_matrix']
        model_specific_cols = []
        
        # Find model-specific columns
        for col in cleaned_df.columns:
            if col.startswith('ot_') or col.startswith('pd_'):
                model_specific_cols.append(col)
        
        all_param_cols = basic_param_cols + model_specific_cols
        final_cols = important_cols + [col for col in all_param_cols if col in cleaned_df.columns]
        
        result_df = cleaned_df[final_cols]
        
        # Add rank column
        result_df.insert(0, 'rank', range(1, len(result_df) + 1))
        
        # Save to CSV
        result_df.to_csv(filepath, index=False)
        print(f"Top {top_n} results saved to {filepath}")
        
        return result_df
    
    def save_complete_analysis(self, top_n: int = 10):
        """Save complete analysis: pickle, text report, and CSV"""
        
        if self.study is None:
            raise ValueError("No study found. Run optimize() first.")
        
        # Save pickle file
        self.save_results()
        
        # Save analysis report
        self.save_analysis_report()
        
        # Save top N CSV
        df_top = self.save_top_results_csv(top_n=top_n)
        
        print(f"\nComplete analysis saved for study: {self.study_name}")
        print(f"Output directory: {self.output_path}")
        print("Files created:")
        print(f"  - {self.study_name}_results_{self.study_date}.pkl (complete study data)")
        print(f"  - {self.study_name}_analysis_{self.study_date}.txt (detailed report)")
        print(f"  - {self.study_name}_top{top_n}_{self.study_date}.csv (top {top_n} configurations)")
        
        return df_top


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Example data (replace with your actual data)
    np.random.seed(42)
    
    X_validation = [f"example text {i}" for i in range(100)]
    y_validation = np.random.choice(['class_0', 'class_1'], size=100).tolist()
    
    kw_samples_validation = []
    for _ in range(100):
        kw_sample = {
            'class_0': [f"example 0 {i}" for i in range(3)],
            'class_1': [f"example 1 {i}" for i in range(3)]
        }
        kw_samples_validation.append(kw_sample)
    
    # Example 1: Single-objective optimization (backward compatible)
    print("=== SINGLE-OBJECTIVE EXAMPLE ===")
    study1, optimizer1 = optimize_scope_bayesian(
        X_validation=X_validation,
        y_validation=y_validation,
        kw_samples_validation=kw_samples_validation,
        objectives='auc_roc',  # Single objective
        n_trials=20,
        timeout=300
    )
    
    # Get best model
    best_model = optimizer1.get_best_model()
    print("Best model ready for use!")
    
    # Example 2: Multi-objective optimization
    print("\n=== MULTI-OBJECTIVE EXAMPLE ===")
    multi_objectives = [
        ObjectiveConfig('auc_roc', OptimizationDirection.MAXIMIZE, weight=0.6),
        ObjectiveConfig('f1_score', OptimizationDirection.MAXIMIZE, weight=0.4)
    ]
    
    study2, optimizer2 = optimize_scope_bayesian(
        X_validation=X_validation,
        y_validation=y_validation,
        kw_samples_validation=kw_samples_validation,
        objectives=multi_objectives,
        n_trials=20,
        timeout=300
    )
    
    # Get best compromise model
    best_compromise = optimizer2.get_best_compromise_model()
    print("Best compromise model ready for use!")
    
    # Save complete analysis
    optimizer2.save_complete_analysis()
    print("Analysis saved!")