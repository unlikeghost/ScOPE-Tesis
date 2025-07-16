import warnings
import numpy as np
from typing import List, Dict, Tuple


import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from .base import ScOPEOptimizer

from scope.utils.report_generation import make_report

warnings.filterwarnings('ignore')


def create_objective_function(optimizer: ScOPEOptimizer,
                              X_validation: List[str],
                              y_validation: List[str], 
                              kw_samples_validation: List[Dict[str, List[str]]],
                              target_metric: str = 'f1_score'):
    """
    Create objective function for Optuna
    target_metric can be: 'accuracy', 'f1_score', 'auc_roc', 'log_loss', 'combined'
    """
    
    def objective(trial):
        params = {
            'compressor': trial.suggest_categorical('compressor', optimizer.available_compressors),
            'name_distance_function': trial.suggest_categorical('name_distance_function', 
                                                               optimizer.available_distance_functions),
            'use_best_sigma': trial.suggest_categorical('use_best_sigma', [True, False]),
            'use_matching_method': trial.suggest_categorical('use_matching_method', [True, False]),
            'str_separator': trial.suggest_categorical('str_separator', [' ', '\t', '\n', '|', ',']),
        }
        
        try:
            # Create model with suggested configuration (NO training)
            model = optimizer.create_model_from_params(params)
            
            # Evaluate configuration using cross-validation
            scores = optimizer.evaluate_model(
                model=model,
                X_samples=X_validation,
                y_true=y_validation,
                kw_samples_list=kw_samples_validation,
                cv_folds=3
            )
            
            # Target metric
            if target_metric == 'combined':
                final_score = (
                    scores['accuracy'] * 0.25 +
                    scores['f1_score'] * 0.35 +
                    scores['auc_roc'] * 0.35 +
                    (1 - scores['log_loss']) * 0.05
                )
            elif target_metric == 'log_loss':
                final_score = -scores['log_loss']  # Minimize log_loss
            else:
                final_score = scores[target_metric]
            
            return final_score
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return 0.0 if target_metric != 'log_loss' else 10.0
    
    return objective


def optimize_scope_model(X_validation: List[str],
                        y_validation: List[str],
                        kw_samples_validation: List[Dict[str, List[str]]],
                        X_holdout: List[str] = None,
                        y_holdout: List[str] = None,
                        kw_samples_holdout: List[Dict[str, List[str]]] = None,
                        n_trials: int = 50,
                        timeout: int = 1800,
                        target_metric: str = 'auc_roc') -> Tuple[optuna.Study, ScOPEOptimizer]:
    """
    Main function to optimize ScOPE model
    
    Args:
        X_validation: Text samples to evaluate configurations
        y_validation: True labels to calculate metrics
        kw_samples_validation: Context needed for predictions
        X_holdout: Samples for final evaluation (optional)
        y_holdout: Labels for final evaluation (optional)
        kw_samples_holdout: Context for final evaluation (optional)
        n_trials: Number of configurations to try
        timeout: Maximum time in seconds
        target_metric: 'accuracy', 'f1_score', 'auc_roc', 'log_loss', 'combined'
    
    Returns:
        Tuple with Optuna study and optimizer
    """
    
    print("=== BAYESIAN OPTIMIZATION SCOPE ===\n")
    print(f"Validation data: {len(X_validation)} samples")
    print(f"Classes: {sorted(set(y_validation))}")
    print(f"Target metric: {target_metric}")
    print(f"Trials: {n_trials}, Timeout: {timeout}s\n")
    
    unique_classes = set(y_validation)
    if len(unique_classes) != 2:
        raise ValueError(f"Expected binary classification, but found {len(unique_classes)} classes: {unique_classes}")
    
    # Create optimizer
    optimizer = ScOPEOptimizer()
    
    # Create objective function
    objective_func = create_objective_function(
        optimizer=optimizer,
        X_validation=X_validation,
        y_validation=y_validation,
        kw_samples_validation=kw_samples_validation,
        target_metric=target_metric
    )
    
    # Configure Optuna study
    direction = 'maximize' if target_metric != 'log_loss' else 'minimize'
    study = optuna.create_study(
        direction=direction,
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,
            n_ei_candidates=24
        ),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1
        ),
        study_name='scope_optimization'
    )
    
    # Execute optimization
    print("Starting optimization...")
    study.optimize(objective_func, n_trials=n_trials, timeout=timeout)
    
    # Show results
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Best score: {study.best_value:.4f}")
    print("Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Evaluate on holdout set if available
    if X_holdout is not None and y_holdout is not None and kw_samples_holdout is not None:
        print("\n=== HOLDOUT SET EVALUATION ===")
        
        # Create final model with best configuration found
        final_model = optimizer.create_model_from_params(study.best_params)
        
        # Evaluate on holdout set (direct evaluation without CV)
        print("Evaluating on holdout set...")
        
        # Direct evaluation without cross-validation
        unique_classes_list = sorted(list(set(y_holdout)))
        class_to_idx = {unique_classes_list[0]: 0, unique_classes_list[1]: 1}
        
        y_holdout_numeric = np.array([class_to_idx[cls] for cls in y_holdout])
        y_pred_holdout = []
        y_pred_proba_holdout = []
        
        # Generate predictions on holdout set
        for sample, kw_sample in zip(X_holdout, kw_samples_holdout):
            try:
                softmax_probs = final_model.__forward__(sample, kw_sample)
                class_names = sorted(softmax_probs.keys())
                proba_values = [softmax_probs[cls] for cls in class_names]
                predicted_class_idx = np.argmax(proba_values)
                
                y_pred_holdout.append(predicted_class_idx)
                y_pred_proba_holdout.append(proba_values)
                
            except Exception as e:
                print(f"Error in holdout prediction: {e}")
                y_pred_holdout.append(0)  # Default prediction
                y_pred_proba_holdout.append([0.5, 0.5])
        
        y_pred_holdout = np.array(y_pred_holdout)
        y_pred_proba_holdout = np.array(y_pred_proba_holdout)
        
        # Calculate metrics using make_report
        try:
            holdout_report = make_report(y_holdout_numeric, y_pred_holdout, y_pred_proba_holdout)
            holdout_scores = {
                'accuracy': holdout_report['acc'],
                'f1_score': holdout_report['f1_score'],
                'auc_roc': holdout_report['auc_roc'],
                'log_loss': holdout_report['log_loss']
            }
        except Exception as e:
            print(f"Error calculating holdout metrics: {e}")
            holdout_scores = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'auc_roc': 0.5,
                'log_loss': 1.0
            }
        
        print("Holdout set results:")
        for metric, score in holdout_scores.items():
            print(f"  {metric}: {score:.4f}")
            
        # Generate detailed report using make_report
        print("\n=== DETAILED HOLDOUT REPORT ===")
        try:
            print(f"Accuracy: {holdout_report['acc']:.4f}")
            print(f"F1 Score: {holdout_report['f1_score']:.4f}")
            print(f"AUC ROC: {holdout_report['auc_roc']:.4f}")
            print(f"Log Loss: {holdout_report['log_loss']:.4f}")
            print(f"Confusion Matrix:\n{holdout_report['confusion_matrix']}")
            
        except Exception as e:
            print(f"Error generating detailed report: {e}")
    
    return study, optimizer


def analyze_optimization_results(study: optuna.Study):
    """Analyze optimization results"""
    
    print("\n=== DETAILED ANALYSIS ===")
    
    # Basic statistics
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    
    print(f"Completed trials: {completed_trials}")
    print(f"Pruned trials: {pruned_trials}")
    print(f"Failed trials: {failed_trials}")
    
    # Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter importance:")
        for param, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {param}: {importance:.4f}")
    except:
        print("Could not calculate parameter importance")
    
    # DataFrame with results
    df_results = study.trials_dataframe()
    if not df_results.empty:
        print("\nTop 5 configurations:")
        columns_to_show = ['value', 'params_compressor', 'params_name_distance_function', 
                          'params_use_best_sigma', 'params_use_matching_method']
        available_columns = [col for col in columns_to_show if col in df_results.columns]
        top_trials = df_results.nlargest(5, 'value')[available_columns]
        print(top_trials)
    
    return df_results


def save_optimization_results(study: optuna.Study, filename: str = "scope_optimization_results.pkl"):
    """Save optimization results"""
    import pickle
    
    results = {
        'study': study,
        'best_params': study.best_params,
        'best_value': study.best_value,
        'trials_dataframe': study.trials_dataframe()
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to {filename}")


def load_optimization_results(filename: str = "scope_optimization_results.pkl"):
    """Load optimization results"""
    import pickle
    
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    return results


def compare_configurations(optimizer: ScOPEOptimizer,
                         configs_to_compare: List[Dict[str, any]],
                         X_test: List[str],
                         y_test: List[str],
                         kw_samples_test: List[Dict[str, List[str]]]):
    """Compare multiple ScOPE configurations"""
    
    print("=== CONFIGURATION COMPARISON ===\n")
    
    results = []
    for i, config in enumerate(configs_to_compare):
        print(f"Evaluating configuration {i+1}/{len(configs_to_compare)}...")
        
        try:
            model = optimizer.create_model_from_params(config)
            
            # Direct evaluation without CV for comparison
            unique_classes = sorted(list(set(y_test)))
            class_to_idx = {unique_classes[0]: 0, unique_classes[1]: 1}
            
            y_test_numeric = np.array([class_to_idx[cls] for cls in y_test])
            y_pred = []
            y_pred_proba = []
            
            for sample, kw_sample in zip(X_test, kw_samples_test):
                try:
                    softmax_probs = model.__forward__(sample, kw_sample)
                    class_names = sorted(softmax_probs.keys())
                    proba_values = [softmax_probs[cls] for cls in class_names]
                    predicted_class_idx = np.argmax(proba_values)
                    
                    y_pred.append(predicted_class_idx)
                    y_pred_proba.append(proba_values)
                except:
                    y_pred.append(0)
                    y_pred_proba.append([0.5, 0.5])
            
            y_pred = np.array(y_pred)
            y_pred_proba = np.array(y_pred_proba)
            
            # Calculate metrics
            report = make_report(y_test_numeric, y_pred, y_pred_proba)
            scores = {
                'accuracy': report['acc'],
                'f1_score': report['f1_score'],
                'auc_roc': report['auc_roc'],
                'log_loss': report['log_loss']
            }
            
        except Exception as e:
            print(f"Error evaluating config {i+1}: {e}")
            scores = {'accuracy': 0.0, 'f1_score': 0.0, 'auc_roc': 0.5, 'log_loss': 1.0}
        
        results.append({
            'config': config,
            'scores': scores
        })
    
    # Show comparison
    print("\n=== COMPARISON RESULTS ===")
    for i, result in enumerate(results):
        print(f"\nConfiguration {i+1}:")
        config = result['config']
        scores = result['scores']
        
        print(f"  Compressor: {config['compressor']}")
        print(f"  Distance function: {config['name_distance_function']}")
        print(f"  Use best sigma: {config['use_best_sigma']}")
        print(f"  Use matching method: {config['use_matching_method']}")
        print(f"  Accuracy: {scores['accuracy']:.4f}")
        print(f"  F1-score: {scores['f1_score']:.4f}")
        print(f"  AUC ROC: {scores['auc_roc']:.4f}")
    
    return results


# ========================================
# USAGE EXAMPLE
# ========================================

if __name__ == '__main__':
    print("=== SCOPE OPTIMIZATION EXAMPLE ===\n")
    
    # Example data
    np.random.seed(42)
    
    # Simulate binary data for validation
    X_validation = [f"example text {i}" for i in range(100)]
    y_validation = np.random.choice(['class_0', 'class_1'], size=100).tolist()
    
    kw_samples_validation = []
    for _ in range(100):
        kw_sample = {
            'class_0': [f"example 0 {i}" for i in range(3)],
            'class_1': [f"example 1 {i}" for i in range(3)]
        }
        kw_samples_validation.append(kw_sample)
    
    # Simulate holdout data for final evaluation
    X_holdout = [f"holdout text {i}" for i in range(20)]
    y_holdout = np.random.choice(['class_0', 'class_1'], size=20).tolist()
    
    kw_samples_holdout = []
    for _ in range(20):
        kw_sample = {
            'class_0': [f"holdout example 0 {i}" for i in range(3)],
            'class_1': [f"holdout example 1 {i}" for i in range(3)]
        }
        kw_samples_holdout.append(kw_sample)
    
    print(f"Validation: {len(X_validation)} samples")
    print(f"Holdout: {len(X_holdout)} samples")
    print(f"Classes: {set(y_validation)}")
    
    # Execute optimization
    study, optimizer = optimize_scope_model(
        X_validation=X_validation,
        y_validation=y_validation,
        kw_samples_validation=kw_samples_validation,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        kw_samples_holdout=kw_samples_holdout,
        n_trials=20,
        timeout=600,
        target_metric='auc_roc'
    )
    
    # Analyze results
    df_results = analyze_optimization_results(study)
    
    print("\n=== FINAL OPTIMIZED MODEL ===")
    print("Optimal configuration found:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Create model with optimal configuration for future use
    final_optimized_model = optimizer.create_model_from_params(study.best_params)
    print("\nOptimized model ready to use!")
    print("Usage: optimized_model.__forward__(sample, kw_sample)")
    
    # Save results
    save_optimization_results(study, "scope_optimization_example.pkl")
    
    print("\n=== AVAILABLE METRICS ===")
    print("- 'accuracy': Overall accuracy")
    print("- 'f1_score': Binary F1 score")
    print("- 'auc_roc': Area under ROC curve")
    print("- 'log_loss': Logarithmic loss")
    print("- 'combined': Weighted combination of metrics")
