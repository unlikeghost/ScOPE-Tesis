import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import deepchem as dc
from sklearn.model_selection import train_test_split

from scope.utils import ScOPEOptimizerBayesian
from scope.utils.report_generation import make_report
from scope.utils.sample_generation import SampleGenerator


seed: int = 42
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams['figure.max_open_warning'] = 0
np.random.seed(seed)

STUDY_NAME: str = 'Clintox'
TEST_SAMPLES:list = 5
TRIALS: int = 2500
CVFOLDS: int = 5

RESULTS_PATH: str = os.path.join('results')
ANALYSYS_RESULTS_PATH: str = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), 'Optimization')
EVALUATION_RESULTS_PATH: str = os.path.join(RESULTS_PATH, str(TEST_SAMPLES), 'Evaluation')

SMILES_COLUMN: str = 'smiles'
LABEL_COLUMN: str = 'fda_approved'



def calculate_timeout_with_cv(sample_size: int) -> int:
    base_samples = 3
    base_cv_folds = 3
    base_time_per_trial = 0.73
    
    sample_factor = (sample_size / base_samples) ** 2  # O(n²)
    cv_factor = CVFOLDS / base_cv_folds  # O(k)
    
    estimated_time_per_trial = base_time_per_trial * sample_factor * cv_factor
    
    min_trials = int(TRIALS * .70)
        
    calculated_timeout = int(estimated_time_per_trial * min_trials)
    
    return calculated_timeout


tasks, datasets, _ = dc.molnet.load_clintox(featurizer='Raw')
train, valid, test = datasets

full_dataset = dc.data.DiskDataset.merge([train, valid, test])

smiles_list = full_dataset.ids
labels = full_dataset.y.astype(int)

print("Número de muestras:", len(smiles_list))
print("Forma de etiquetas:", labels.shape)
print("Tareas:", tasks)


df = pd.DataFrame({
    "smiles": smiles_list,
    "fda_approved": labels[:, 0],
    "ct_tox": labels[:, 1]
})


print(df.head())
print("\nDistribución de clases - FDA_APPROVED:")
print(df["fda_approved"].value_counts())
print("\nDistribución de clases - CT_TOX:")
print(df["ct_tox"].value_counts())


x_test, x_search, y_test, y_search =  train_test_split(
    df['smiles'].values,
    df[LABEL_COLUMN].values,
    train_size=.70,
    random_state=seed,
    stratify=df[LABEL_COLUMN].values
)

print(x_test.shape, y_test.shape, x_search.shape, y_search.shape)


search_generator = SampleGenerator(
    data=x_search,
    labels=y_search,
    seed=seed,
)

timeout: int = calculate_timeout_with_cv(
    sample_size=TEST_SAMPLES
)

optimizer = ScOPEOptimizerBayesian(
    free_cpu=1,
    n_trials=TRIALS,
    timeout=timeout,
    target_metric='auc_roc',
    study_name=f'{STUDY_NAME}_Samples_{TEST_SAMPLES}',
    output_path=ANALYSYS_RESULTS_PATH,
    cv_folds=CVFOLDS
)
    
all_x = []
all_y = []
all_kw = []

for x_search_i, y_search_i, search_kw_samples_i in search_generator.generate(num_samples=TEST_SAMPLES):
    all_x.append(x_search_i)
    all_y.append(y_search_i)
    all_kw.append(search_kw_samples_i)


study = optimizer.optimize(all_x, all_y, all_kw)

optimizer.save_complete_analysis(top_n=1000)

best_model = optimizer.get_best_model()

test_generator = SampleGenerator(
    data=x_test,
    labels=y_test,
    seed=seed,
)

all_y_true = []
all_y_predicted = []
all_y_probas = []

for x_test_i, y_test_i, test_kw_samples_i in test_generator.generate(num_samples=TEST_SAMPLES):
    
    
    softmax_probs = list(best_model(
        list_samples=x_test_i,
        list_kw_samples=test_kw_samples_i
    ))[0]['softmax']
    
    class_names = sorted(softmax_probs.keys())
    proba_values = [softmax_probs[cls] for cls in class_names]
    
    predicted_class_idx = np.argmax(proba_values)
                        
    # print(predicted_class_idx, y_test_i, proba_values)
    all_y_true.append(
        y_test_i
    )
    
    all_y_predicted.append(
        predicted_class_idx
    )
    
    all_y_probas.append(
        proba_values
    )
    
results = make_report(
    y_true=all_y_true,
    y_pred=all_y_predicted,
    y_pred_proba=all_y_probas,
    save_path=EVALUATION_RESULTS_PATH
)

print(results)