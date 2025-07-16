import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from scope.utils.param_search.bayesian import optimize_scope_model, analyze_optimization_results
from scope.samples.sample_generator import generate_samples


FILE_NAME: str = 'clintox'
FILE_PATH: str = os.path.join('data', 'dataset', f'{FILE_NAME}.csv')
RESULTS_PATH: str = os.path.join('results', 'results')

SMILES_COLUMN: str = 'smiles'
LABEL_COLUMN: str = 'CT_TOX'

MIN_SAMPLES: int = 10
MAX_SAMPLES: int = 11

dataframe: pd.DataFrame = pd.read_csv(FILE_PATH)
X: np.ndarray = dataframe[SMILES_COLUMN].values
Y: np.ndarray = dataframe[LABEL_COLUMN].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

for num_samples in range(MIN_SAMPLES, MAX_SAMPLES + 1, 10):
    X_validation = []
    y_validation = []
    kw_samples_validation = []
    
    for index, (current_x, current_y, current_kw_samples) in enumerate(generate_samples(data=x_train, labels=y_train, num_samples=num_samples)):
        X_validation.append(current_x)
        y_validation.append(current_y),
        kw_samples_validation.append(current_kw_samples)
    
    study, optimizer = optimize_scope_model(
        X_validation=X_validation,
        y_validation=y_validation,
        kw_samples_validation=kw_samples_validation,
        n_trials=20,
        timeout=600,
        target_metric='auc_roc'
    )
    
    df_results = analyze_optimization_results(study)

    print("\n=== FINAL OPTIMIZED MODEL ===")
    print("Optimal configuration found:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")

    # Create model with optimal configuration for future use
    final_optimized_model = optimizer.create_model_from_params(study.best_params)
    print("\nOptimized model ready to use!")
    print("Usage: optimized_model.__forward__(sample, kw_sample)")
    
    
    for index, (current_x, current_y, current_kw_samples) in enumerate(generate_samples(data=x_test, labels=y_test, num_samples=num_samples)):

        prediction = np.argmax(
                list(final_optimized_model.__forward__(
                current_x, 
                current_kw_samples
            ).values())
        )
        
        print(f"Prediction: {prediction}, {current_y}")

    break