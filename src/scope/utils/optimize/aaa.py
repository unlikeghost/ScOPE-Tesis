import os
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
    """Bayesian optimization for ScOPE models using Optuna."""
    
    def __init__(self, 
                 parameter_space: Optional[ParameterSpace] = None,
                 free_cpu: int = 0,
                 n_trials: int = 50,
                 timeout: int = 1800,
                 target_metric: str = 'auc_roc',
                 random_seed: int = 42,
                 cv_folds: int = 3,
                 study_name: str = "scope_optimization",
                 output_path: str = "./results"):
        """Initialize the Bayesian optimizer"""
        super().__init__(parameter_space, free_cpu, random_seed=random_seed)
        
        self.n_trials = n_trials
        self.timeout = timeout
        self.target_metric = target_metric
        self.study_name = study_name
        self.output_path = output_path
        self.cv_folds = cv_folds
        
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
                    normalized_log_loss = 1 / (1 + scores['log_loss'])
                    final_score = (
                        scores['auc_roc'] * 0.50 +
                        scores['f1_score'] * 0.40 +
                        normalized_log_loss * 0.10
                   )
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
        print(f"Trials: {self.n_trials}, Timeout: {self.timeout}s")
        print(f"CV Folds: {self.cv_folds}\n")
        
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

    # Métodos de análisis y guardado (omitidos por brevedad pero corregidos)
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
                              'params_use_best_sigma', 'params_symetric_matrix' 'params_model_type']
            
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
            f.write(f"Number of trials: {self.n_trials}\n")
            f.write(f"Timeout: {self.timeout} seconds\n")
            f.write(f"CV folds: {self.cv_folds}\n")
            f.write(f"Random seed: {self.random_seed}\n")
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
            
            # Parameter importance
            try:
                importances = optuna.importance.get_param_importances(self.study)
                f.write("PARAMETER IMPORTANCE:\n")
                f.write("-" * 30 + "\n")
                
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
                
                # Write basic parameters
                if basic_params:
                    f.write("Basic Parameters:\n")
                    for param, importance in sorted(basic_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write ScOPE-OT parameters
                if ot_params:
                    f.write("ScOPE-OT Parameters:\n")
                    for param, importance in sorted(ot_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write ScOPE-PD parameters
                if pd_params:
                    f.write("ScOPE-PD Parameters:\n")
                    for param, importance in sorted(pd_params.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {importance:.6f}\n")
                    f.write("\n")
                
                # Write overall ranking
                f.write("Overall Ranking (All Parameters):\n")
                for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {param}: {importance:.6f}\n")
                f.write("\n")
                
            except Exception as e:
                f.write(f"PARAMETER IMPORTANCE: Could not calculate ({str(e)})\n\n")
            
            # Top 10 configurations
            df_results = self.study.trials_dataframe()
            if not df_results.empty:
                top_10 = df_results.nlargest(10, 'value')
                f.write("TOP 10 CONFIGURATIONS:\n")
                f.write("-" * 30 + "\n")
                
                for i, (idx, row) in enumerate(top_10.iterrows(), 1):
                    f.write(f"\nRank {i} (Trial {row['number']}):\n")
                    f.write(f"  Score: {row['value']:.6f}\n")
                    
                    # Extract parameters
                    param_cols = [col for col in df_results.columns if col.startswith('params_')]
                    for param_col in param_cols:
                        param_name = param_col.replace('params_', '')
                        param_value = row[param_col]
                        if param_name == 'string_separator':
                            param_value = repr(param_value)
                        elif pd.isna(param_value):
                            param_value = 'None'
                        f.write(f"  {param_name}: {param_value}\n")
        
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