import warnings
import numpy as np
from typing import List, Dict, Any
from sklearn.model_selection import StratifiedKFold

from scope import ScOPE
from scope.utils.report_generation import make_report

warnings.filterwarnings('ignore')


class ScOPEOptimizer:
    
    def __init__(self, available_compressors: List[str] = None,
                 available_distance_functions: List[str] = None):
        
        self.available_compressors = available_compressors or ['gzip', 'bz2', 'lz77', 'zstandard']
        self.available_distance_functions = available_distance_functions or [
            'ncd', 'cdm', 'clm', 'mse'
        ]
        
    def create_model_from_params(self, params: Dict[str, Any]) -> ScOPE:
        
        return ScOPE(
            compressor=params['compressor'],
            name_distance_function=params['name_distance_function'],
            use_best_sigma=params['use_best_sigma'],
            str_separator=params['str_separator'],
            use_matching_method=params['use_matching_method']
        )
    
    def evaluate_model(self, 
                      model: ScOPE,
                      X_samples: List[str],
                      y_true: List[str],
                      kw_samples_list: List[Dict[str, List[str]]],
                      cv_folds: int = 3) -> Dict[str, float]:

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
            raise ValueError(f"Se esperaban exactamente 2 clases, pero se encontraron {len(unique_classes)}: {unique_classes}")
        
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
                        softmax_probs = model.__forward__(sample, kw_sample)
                        
                        class_names = sorted(softmax_probs.keys())
                        proba_values = [softmax_probs[cls] for cls in class_names]
                        predicted_class_idx = np.argmax(proba_values)
                        
                        y_pred.append(predicted_class_idx)
                        y_pred_proba.append(proba_values)
                        
                    except Exception as e:
                        print(f"Error en predicción individual: {e}")
                        # Predicción aleatoria en caso de error
                        random_pred = np.random.randint(0, 2)
                        y_pred.append(random_pred)
                        
                        # Probabilidades aleatorias
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
